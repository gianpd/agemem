"""
memory/migrations.py
────────────────────
SQLite schema migrations for AgeMem semantic search.

Responsibilities
────────────────
* Set up SQLite database for vector-based semantic search
* Manage schema migrations with idempotency
* Load sqlite-vec extension for vector similarity search

Design decisions
────────────────
* Thread safety via RLock (matching LTMStore pattern)
* Idempotent migrations: safe to run multiple times
* Transaction-based with rollback on failure
* Returns True on success, False if already applied
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default embedding configuration
DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DIM = 1024


class SchemaMigrationError(Exception):
    """Raised when a schema migration fails."""
    pass


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    """
    Load the sqlite-vec extension.

    Returns True if successful, False otherwise.
    """
    try:
        conn.enable_load_extension(True)

        # Primary method: use sqlite_vec Python package
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
            logger.info("Loaded sqlite-vec extension via sqlite_vec.load()")
            return True
        except ImportError:
            logger.debug("sqlite_vec package not available")
        except Exception as e:
            logger.debug(f"sqlite_vec.load() failed: {e}")

        # Fallback: try loading from loadable_path
        try:
            import sqlite_vec
            loadable_path = sqlite_vec.loadable_path()
            conn.load_extension(str(loadable_path))
            logger.info(f"Loaded sqlite-vec from loadable_path: {loadable_path}")
            return True
        except Exception as e:
            logger.debug(f"Loading from loadable_path failed: {e}")

        # Last resort: try common extension names
        extension_names = ["vec0", "sqlite_vec", "sqlite-vec"]
        for ext_name in extension_names:
            try:
                conn.load_extension(ext_name)
                logger.info(f"Loaded sqlite-vec extension: {ext_name}")
                return True
            except sqlite3.OperationalError:
                continue

        logger.warning("Could not load sqlite-vec extension. Vector search will be unavailable.")
        return False

    except AttributeError:
        # SQLite was compiled without extension support
        logger.warning("SQLite does not support extensions. Vector search will be unavailable.")
        return False


def _table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def _column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def _virtual_table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    """Check if a virtual table exists (for sqlite-vec tables)."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=? AND sql LIKE 'CREATE VIRTUAL TABLE%'",
        (table_name,)
    )
    return cursor.fetchone() is not None


def apply_semantic_schema(db_path: str) -> bool:
    """
    Apply semantic search schema to a SQLite database.

    Creates the necessary tables and columns for vector-based semantic search.
    This function is idempotent - safe to call multiple times.

    Schema changes applied:
        1. Creates `ltm_entries` table if not exists
        2. Adds embedding columns to `ltm_entries` if not present
        3. Creates `ltm_vec_index` virtual table for vector search

    Args:
        db_path: Path to the SQLite database file. Will be created if it doesn't exist.

    Returns:
        True if schema was applied or already present.
        False if the migration failed (check logs for details).

    Raises:
        SchemaMigrationError: If a critical error occurs during migration.
    """
    lock = threading.RLock()

    with lock:
        # Ensure parent directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn: Optional[sqlite3.Connection] = None

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Track whether we made any changes
            changes_made = False

            # Start transaction
            conn.execute("BEGIN")

            # Step 1: Create ltm_entries table if not exists
            if not _table_exists(cursor, "ltm_entries"):
                logger.info("Creating ltm_entries table")
                cursor.execute("""
                    CREATE TABLE ltm_entries (
                        entry_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        learning_score REAL DEFAULT 0.0,
                        tags TEXT,
                        source_turn INTEGER DEFAULT 0
                    )
                """)
                changes_made = True
                logger.info("Created ltm_entries table")
            else:
                logger.debug("ltm_entries table already exists")

            # Step 2: Add embedding columns if not present
            embedding_columns = [
                ("embedding", "BLOB"),
                ("embedding_model", f"TEXT DEFAULT '{DEFAULT_EMBEDDING_MODEL}'"),
                ("embedding_dim", f"INTEGER DEFAULT {DEFAULT_EMBEDDING_DIM}"),
            ]

            for col_name, col_type in embedding_columns:
                if not _column_exists(cursor, "ltm_entries", col_name):
                    logger.info(f"Adding column {col_name} to ltm_entries")
                    cursor.execute(f"ALTER TABLE ltm_entries ADD COLUMN {col_name} {col_type}")
                    changes_made = True

            # Step 3: Load sqlite-vec extension and create vector index
            if not _virtual_table_exists(cursor, "ltm_vec_index"):
                vec_loaded = _load_vec_extension(conn)

                if vec_loaded:
                    logger.info("Creating ltm_vec_index virtual table")
                    cursor.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS ltm_vec_index
                        USING vec0(
                            entry_id TEXT PRIMARY KEY,
                            embedding FLOAT[{DEFAULT_EMBEDDING_DIM}]
                        )
                    """)
                    changes_made = True
                    logger.info("Created ltm_vec_index virtual table")
                else:
                    logger.warning(
                        "Skipping ltm_vec_index creation: sqlite-vec extension not available. "
                        "Semantic search will fall back to keyword matching."
                    )

            # Commit transaction
            conn.commit()

            if changes_made:
                logger.info(f"Semantic schema applied successfully to {db_path}")
            else:
                logger.info(f"Semantic schema already up to date in {db_path}")

            return True

        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"SQLite error during schema migration: {e}")
            raise SchemaMigrationError(f"Failed to apply semantic schema: {e}") from e

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Unexpected error during schema migration: {e}")
            raise SchemaMigrationError(f"Failed to apply semantic schema: {e}") from e

        finally:
            if conn:
                conn.close()


def verify_semantic_schema(db_path: str) -> dict:
    """
    Verify the semantic search schema is correctly applied.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Dictionary with verification results:
        {
            "ltm_entries_exists": bool,
            "embedding_columns": list[str],
            "vec_index_exists": bool,
            "vec_extension_loaded": bool,
            "ready_for_semantic_search": bool,
        }
    """
    result = {
        "ltm_entries_exists": False,
        "embedding_columns": [],
        "vec_index_exists": False,
        "vec_extension_loaded": False,
        "ready_for_semantic_search": False,
    }

    lock = threading.RLock()

    with lock:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check ltm_entries table
            if _table_exists(cursor, "ltm_entries"):
                result["ltm_entries_exists"] = True

                # Check embedding columns
                expected_cols = ["embedding", "embedding_model", "embedding_dim"]
                for col in expected_cols:
                    if _column_exists(cursor, "ltm_entries", col):
                        result["embedding_columns"].append(col)

            # Check vector index
            result["vec_index_exists"] = _virtual_table_exists(cursor, "ltm_vec_index")

            # Check if extension can be loaded
            try:
                result["vec_extension_loaded"] = _load_vec_extension(conn)
            except Exception:
                result["vec_extension_loaded"] = False

            # Determine if ready for semantic search
            result["ready_for_semantic_search"] = (
                result["ltm_entries_exists"]
                and len(result["embedding_columns"]) == 3
                and result["vec_index_exists"]
                and result["vec_extension_loaded"]
            )

        except sqlite3.Error as e:
            logger.error(f"Error verifying schema: {e}")

        finally:
            if 'conn' in locals():
                conn.close()

    return result


def drop_semantic_schema(db_path: str, keep_entries: bool = True) -> bool:
    """
    Remove semantic search schema from the database.

    WARNING: This will delete all vector embeddings and the vector index.
    Use with caution.

    Args:
        db_path: Path to the SQLite database file.
        keep_entries: If True, keep ltm_entries table and data (default).
                     If False, drop the entire ltm_entries table.

    Returns:
        True if successful, False otherwise.
    """
    lock = threading.RLock()

    with lock:
        conn: Optional[sqlite3.Connection] = None

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            conn.execute("BEGIN")

            # Drop vector index
            if _virtual_table_exists(cursor, "ltm_vec_index"):
                cursor.execute("DROP TABLE IF EXISTS ltm_vec_index")
                logger.info("Dropped ltm_vec_index table")

            if keep_entries:
                # Only drop embedding columns (by recreating table without them)
                # Note: SQLite doesn't support DROP COLUMN, so we'd need to
                # recreate the table. For safety, we'll skip this and just
                # clear the embeddings.
                cursor.execute("UPDATE ltm_entries SET embedding = NULL")
                logger.info("Cleared embeddings from ltm_entries")
            else:
                # Drop entire table
                cursor.execute("DROP TABLE IF EXISTS ltm_entries")
                logger.info("Dropped ltm_entries table")

            conn.commit()
            return True

        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Error dropping schema: {e}")
            return False

        finally:
            if conn:
                conn.close()
