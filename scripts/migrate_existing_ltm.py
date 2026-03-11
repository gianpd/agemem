#!/usr/bin/env python3
"""
scripts/migrate_existing_ltm.py
-------------------------------
Migrate existing LTM entries to semantic search by generating embeddings.

This script:
1. Loads existing LTM entries from JSON
2. Generates embeddings for entries without embeddings
3. Inserts embeddings into the vector index
4. Commits per batch for safety

Usage:
    python scripts/migrate_existing_ltm.py [--batch-size 32] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Migrate LTM entries to semantic search")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of entries to process per batch (default: 32)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--ltm-path", type=str, default=None,
        help="Path to LTM JSON file (default: agent_memory/ltm_store.json)"
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to SQLite database (default: agent_memory/ltm_semantic.db)"
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    ltm_path = Path(args.ltm_path) if args.ltm_path else project_root / "agent_memory" / "ltm_store.json"
    db_path = Path(args.db_path) if args.db_path else project_root / "agent_memory" / "ltm_semantic.db"

    print(f"LTM JSON path: {ltm_path}")
    print(f"SQLite path: {db_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Check if LTM file exists
    if not ltm_path.exists():
        print(f"No LTM file found at {ltm_path}")
        print("Nothing to migrate.")
        return 0

    # Load LTM entries
    try:
        data = json.loads(ltm_path.read_text())
        entries = data if isinstance(data, list) else []
        print(f"Loaded {len(entries)} entries from LTM")
    except Exception as e:
        print(f"ERROR: Failed to load LTM file: {e}")
        return 1

    if not entries:
        print("No entries to migrate.")
        return 0

    # Initialize SQLite database
    print("\nInitializing SQLite database...")
    try:
        from core.db_migrations import apply_semantic_schema
        apply_semantic_schema(str(db_path))
        print(f"Schema applied to {db_path}")
    except Exception as e:
        print(f"ERROR: Failed to initialize database: {e}")
        return 1

    # Open database connection
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(db)
    except ImportError:
        print("ERROR: sqlite-vec not installed. Run: uv pip install sqlite-vec")
        return 1

    # Check which entries already have embeddings
    from memory.vector_index import get_embedding_count, entry_exists

    existing_count = get_embedding_count(db)
    print(f"Existing embeddings in vector index: {existing_count}")

    # Initialize embedding model
    print("\nLoading embedding model...")
    try:
        from memory.embedding import EmbeddingModule
        model = EmbeddingModule.get_instance()
        print(f"Model loaded: {model.MODEL_NAME}")
        print(f"Cache path: {model.cache_path}")
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        return 1

    # Process entries in batches
    print(f"\nProcessing {len(entries)} entries...")
    processed = 0
    skipped = 0
    errors = 0

    start_time = time.time()

    for i, entry in enumerate(entries):
        entry_id = entry.get("entry_id")
        content = entry.get("content", "")

        if not entry_id or not content:
            print(f"  Skipping entry {i}: missing entry_id or content")
            skipped += 1
            continue

        # Skip if already has embedding
        if entry_exists(db, entry_id):
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [DRY-RUN] Would embed: {entry_id[:12]}... ({len(content)} chars)")
            processed += 1
            continue

        try:
            # Generate embedding
            embedding = model.embed_text(content)

            # Insert into vector index
            from memory.vector_index import insert_embedding
            insert_embedding(db, entry_id, embedding)

            # Also upsert to ltm_entries table
            import json as json_mod
            db.execute("""
                INSERT OR REPLACE INTO ltm_entries
                (entry_id, content, created_at, updated_at, access_count, learning_score, tags, source_turn)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                content,
                entry.get("created_at", time.time()),
                entry.get("updated_at", time.time()),
                entry.get("access_count", 0),
                entry.get("learning_score", 0.0),
                json_mod.dumps(entry.get("tags", [])),
                entry.get("source_turn", 0),
            ))

            processed += 1

            # Commit per batch
            if processed % args.batch_size == 0:
                db.commit()
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  Progress: {processed}/{len(entries)} ({rate:.1f} entries/sec)")

        except Exception as e:
            print(f"  ERROR processing {entry_id}: {e}")
            errors += 1

    # Final commit
    if not args.dry_run:
        db.commit()

    # Summary
    elapsed = time.time() - start_time
    print(f"\nMigration complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s")

    # Verify
    final_count = get_embedding_count(db)
    print(f"  Total embeddings: {final_count}")

    db.close()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())