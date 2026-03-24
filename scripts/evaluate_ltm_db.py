#!/usr/bin/env python3
"""
Quick evaluation script for ltm_semantic.db

Usage:
    python scripts/evaluate_ltm_db.py [--db-path PATH]

Shows:
- Schema overview
- Entry statistics
- Embedding coverage
- Tag distribution
- Learning score distribution
- Recent activity
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec extension for vector table access."""
    try:
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
            return True
        except ImportError:
            pass
        # Fallback methods
        for ext_name in ["vec0", "sqlite_vec", "sqlite-vec"]:
            try:
                conn.load_extension(ext_name)
                return True
            except sqlite3.OperationalError:
                continue
        return False
    except AttributeError:
        return False


def get_db_info(db_path: str) -> dict[str, Any]:
    """Collect comprehensive database information."""
    info = {
        "db_path": db_path,
        "db_size_mb": 0,
        "tables": [],
        "ltm_entries": {},
        "vec_index": {},
    }

    conn = sqlite3.connect(db_path)
    _load_vec_extension(conn)  # Load extension for vector table access
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # File size
    info["db_size_mb"] = Path(db_path).stat().st_size / (1024 * 1024)

    # List all tables
    cursor.execute("""
        SELECT name, type, sql
        FROM sqlite_master
        WHERE type IN ('table', 'virtual table')
        ORDER BY name
    """)
    info["tables"] = [dict(row) for row in cursor.fetchall()]

    # ltm_entries stats
    cursor.execute("SELECT COUNT(*) as total FROM ltm_entries")
    info["ltm_entries"]["total_entries"] = cursor.fetchone()["total"]

    if info["ltm_entries"]["total_entries"] > 0:
        # Embedding coverage
        cursor.execute("""
            SELECT
                COUNT(*) as with_embedding,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM ltm_entries) as pct
            FROM ltm_entries
            WHERE embedding IS NOT NULL
        """)
        row = cursor.fetchone()
        info["ltm_entries"]["embeddings"] = {
            "count": row["with_embedding"],
            "percentage": round(row["pct"], 1),
        }

        # Date range
        cursor.execute("""
            SELECT
                MIN(created_at) as oldest,
                MAX(created_at) as newest
            FROM ltm_entries
        """)
        row = cursor.fetchone()
        info["ltm_entries"]["date_range"] = {
            "oldest": datetime.fromtimestamp(row["oldest"]).isoformat() if row["oldest"] else None,
            "newest": datetime.fromtimestamp(row["newest"]).isoformat() if row["newest"] else None,
        }

        # Learning score distribution
        cursor.execute("""
            SELECT
                AVG(learning_score) as avg_score,
                MIN(learning_score) as min_score,
                MAX(learning_score) as max_score,
                SUM(CASE WHEN learning_score >= 0.5 THEN 1 ELSE 0 END) as high_score_count
            FROM ltm_entries
        """)
        row = cursor.fetchone()
        info["ltm_entries"]["learning_scores"] = {
            "avg": round(row["avg_score"], 3) if row["avg_score"] else 0,
            "min": round(row["min_score"], 3) if row["min_score"] else 0,
            "max": round(row["max_score"], 3) if row["max_score"] else 0,
            "high_score_count": row["high_score_count"],
        }

        # Access patterns
        cursor.execute("""
            SELECT
                AVG(access_count) as avg_access,
                MAX(access_count) as max_access,
                SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) as never_accessed
            FROM ltm_entries
        """)
        row = cursor.fetchone()
        info["ltm_entries"]["access_patterns"] = {
            "avg_access": round(row["avg_access"], 1) if row["avg_access"] else 0,
            "max_access": row["max_access"],
            "never_accessed": row["never_accessed"],
        }

        # Tag distribution
        cursor.execute("""
            SELECT tags, COUNT(*) as cnt
            FROM ltm_entries
            WHERE tags IS NOT NULL AND tags != ''
            GROUP BY tags
            ORDER BY cnt DESC
            LIMIT 10
        """)
        info["ltm_entries"]["top_tags"] = [
            {"tag": row["tags"], "count": row["cnt"]}
            for row in cursor.fetchall()
        ]

        # Source turn distribution
        cursor.execute("""
            SELECT source_turn, COUNT(*) as cnt
            FROM ltm_entries
            GROUP BY source_turn
            ORDER BY source_turn
            LIMIT 10
        """)
        info["ltm_entries"]["source_turns"] = [
            {"turn": row["source_turn"], "count": row["cnt"]}
            for row in cursor.fetchall()
        ]

        # Embedding models in use
        cursor.execute("""
            SELECT embedding_model, COUNT(*) as cnt
            FROM ltm_entries
            WHERE embedding_model IS NOT NULL
            GROUP BY embedding_model
        """)
        info["ltm_entries"]["embedding_models"] = [
            {"model": row["embedding_model"], "count": row["cnt"]}
            for row in cursor.fetchall()
        ]

        # Content length stats
        cursor.execute("""
            SELECT
                AVG(LENGTH(content)) as avg_len,
                MIN(LENGTH(content)) as min_len,
                MAX(LENGTH(content)) as max_len
            FROM ltm_entries
        """)
        row = cursor.fetchone()
        info["ltm_entries"]["content_stats"] = {
            "avg_length": int(row["avg_len"]) if row["avg_len"] else 0,
            "min_length": row["min_len"],
            "max_length": row["max_len"],
        }

        # Recent entries sample
        cursor.execute("""
            SELECT entry_id, substr(content, 1, 80) as content_preview,
                   created_at, learning_score, access_count
            FROM ltm_entries
            ORDER BY created_at DESC
            LIMIT 5
        """)
        info["ltm_entries"]["recent_sample"] = [
            {
                "id": row["entry_id"][:16] + "...",
                "content": row["content_preview"] + ("..." if len(row["content_preview"]) >= 80 else ""),
                "created": datetime.fromtimestamp(row["created_at"]).strftime("%Y-%m-%d %H:%M"),
                "score": row["learning_score"],
                "access": row["access_count"],
            }
            for row in cursor.fetchall()
        ]

        # High value entries (high learning score)
        cursor.execute("""
            SELECT entry_id, substr(content, 1, 60) as content_preview,
                   learning_score, access_count
            FROM ltm_entries
            WHERE learning_score >= 0.7
            ORDER BY learning_score DESC
            LIMIT 5
        """)
        info["ltm_entries"]["high_value"] = [
            {
                "id": row["entry_id"][:16] + "...",
                "content": row["content_preview"] + ("..." if len(row["content_preview"]) >= 60 else ""),
                "score": row["learning_score"],
                "access": row["access_count"],
            }
            for row in cursor.fetchall()
        ]

    # Vector index stats
    try:
        cursor.execute("SELECT COUNT(*) as total FROM ltm_vec_index")
        info["vec_index"]["total_vectors"] = cursor.fetchone()["total"]
    except sqlite3.OperationalError:
        info["vec_index"]["error"] = "Vector index table not found or sqlite-vec not available"

    conn.close()
    return info


def print_report(info: dict[str, Any]) -> None:
    """Print a formatted evaluation report."""
    print("=" * 60)
    print("LTM SEMANTIC DB EVALUATION REPORT")
    print("=" * 60)

    # Database overview
    print(f"\n📁 Database: {info['db_path']}")
    print(f"   Size: {info['db_size_mb']:.2f} MB")
    print(f"   Tables: {len(info['tables'])}")
    for t in info["tables"]:
        table_type = "virtual" if "VIRTUAL" in (t["sql"] or "") else "table"
        print(f"      - {t['name']} ({table_type})")

    # Entry stats
    ltm = info["ltm_entries"]
    print(f"\n📊 ENTRIES: {ltm['total_entries']}")

    if ltm["total_entries"] == 0:
        print("   (No entries in database)")
        return

    # Embeddings
    emb = ltm.get("embeddings", {})
    print(f"\n🔷 EMBEDDINGS")
    print(f"   Coverage: {emb.get('count', 0)} / {ltm['total_entries']} ({emb.get('percentage', 0)}%)")

    models = ltm.get("embedding_models", [])
    if models:
        print("   Models:")
        for m in models:
            print(f"      - {m['model']}: {m['count']} entries")

    # Date range
    dr = ltm.get("date_range", {})
    if dr.get("oldest"):
        print(f"\n📅 DATE RANGE")
        print(f"   Oldest: {dr['oldest']}")
        print(f"   Newest: {dr['newest']}")

    # Learning scores
    ls = ltm.get("learning_scores", {})
    print(f"\n📈 LEARNING SCORES")
    print(f"   Average: {ls.get('avg', 0)}")
    print(f"   Range: {ls.get('min', 0)} - {ls.get('max', 0)}")
    print(f"   High scores (>=0.5): {ls.get('high_score_count', 0)}")

    # Access patterns
    ap = ltm.get("access_patterns", {})
    print(f"\n🔄 ACCESS PATTERNS")
    print(f"   Avg accesses: {ap.get('avg_access', 0)}")
    print(f"   Max accesses: {ap.get('max_access', 0)}")
    print(f"   Never accessed: {ap.get('never_accessed', 0)}")

    # Content stats
    cs = ltm.get("content_stats", {})
    print(f"\n📝 CONTENT STATS")
    print(f"   Avg length: {cs.get('avg_length', 0)} chars")
    print(f"   Range: {cs.get('min_length', 0)} - {cs.get('max_length', 0)} chars")

    # Tags
    tags = ltm.get("top_tags", [])
    if tags:
        print(f"\n🏷️ TOP TAGS")
        for t in tags:
            print(f"   {t['tag']}: {t['count']}")

    # Source turns
    turns = ltm.get("source_turns", [])
    if turns:
        print(f"\n💬 SOURCE TURNS")
        for t in turns[:5]:
            print(f"   Turn {t['turn']}: {t['count']} entries")

    # Recent entries
    recent = ltm.get("recent_sample", [])
    if recent:
        print(f"\n🆕 RECENT ENTRIES")
        for e in recent:
            print(f"   [{e['created']}] {e['content']}")
            print(f"      id={e['id']} score={e['score']} access={e['access']}")

    # High value entries
    high = ltm.get("high_value", [])
    if high:
        print(f"\n⭐ HIGH VALUE ENTRIES (score >= 0.7)")
        for e in high:
            print(f"   {e['content']}")
            print(f"      id={e['id']} score={e['score']} access={e['access']}")

    # Vector index
    vec = info.get("vec_index", {})
    print(f"\n🔍 VECTOR INDEX")
    if "total_vectors" in vec:
        print(f"   Total vectors: {vec['total_vectors']}")
    else:
        print(f"   Status: {vec.get('error', 'Unknown')}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LTM semantic database")
    parser.add_argument(
        "--db-path",
        default="agent_memory/ltm_semantic.db",
        help="Path to the SQLite database",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1

    info = get_db_info(str(db_path))
    print_report(info)
    return 0


if __name__ == "__main__":
    exit(main())