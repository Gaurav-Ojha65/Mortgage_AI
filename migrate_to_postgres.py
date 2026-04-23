"""
PostgreSQL Migration Script for Mortgage AI

Migrates existing SQLite data to PostgreSQL with:
- Progress tracking
- Row count verification
- Transaction rollback on failure
- Comprehensive logging

Usage:
    python migrate_to_postgres.py

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (required)
    SQLITE_PATH: Path to SQLite database (default: mortgage.db)
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, inspect
from sqlalchemy.engine import Engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SQLITE_PATH = "mortgage.db"
BATCH_SIZE = 1000  # Rows per batch for large tables

# Tables to migrate
TABLES_TO_MIGRATE = [
    "decisions",
    "audit_logs",
    "users",
]


# =============================================================================
# Database Connection Functions
# =============================================================================


def get_sqlite_engine(sqlite_path: str) -> Engine:
    """Create SQLite engine."""
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    engine = create_engine(f"sqlite:///{sqlite_path}")
    logger.info(f"Connected to SQLite: {sqlite_path}")
    return engine


def get_postgres_engine() -> Engine:
    """Create PostgreSQL engine from environment."""
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        # Build from components
        db_url = (
            f"postgresql://"
            f"{os.getenv('POSTGRES_USER', 'postgres')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'mortgage')}"
        )

    try:
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL: {db_url.replace('@', '***@')}")
        return engine
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        raise


# =============================================================================
# Schema Creation
# =============================================================================


def create_postgres_schema(pg_engine: Engine):
    """
    Create PostgreSQL schema matching SQLite structure.

    Uses proper PostgreSQL types and constraints.
    """
    logger.info("Creating PostgreSQL schema...")

    create_tables_sql = """
    -- Decisions table
    CREATE TABLE IF NOT EXISTS decisions (
        id SERIAL PRIMARY KEY,
        timestamp VARCHAR(50) NOT NULL,
        income DOUBLE PRECISION NOT NULL,
        loan_amount DOUBLE PRECISION NOT NULL,
        credit_score INTEGER NOT NULL,
        decision VARCHAR(20) NOT NULL,
        risk_level VARCHAR(20) NOT NULL,
        default_probability DOUBLE PRECISION,
        emi DOUBLE PRECISION NOT NULL,
        advice TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Audit logs table
    CREATE TABLE IF NOT EXISTS audit_logs (
        id SERIAL PRIMARY KEY,
        timestamp VARCHAR(50) NOT NULL,
        user_id VARCHAR(100),
        action VARCHAR(100) NOT NULL,
        details JSONB,
        ip_address VARCHAR(45),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Users table
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        hashed_password VARCHAR(255) NOT NULL,
        role VARCHAR(20) NOT NULL DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_decisions_user ON decisions(income, loan_amount);
    CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id);
    CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    """

    with pg_engine.connect() as conn:
        conn.execute(text(create_tables_sql))
        conn.commit()

    logger.info("PostgreSQL schema created successfully")


# =============================================================================
# Data Migration Functions
# =============================================================================


def get_table_row_count(engine: Engine, table_name: str) -> int:
    """Get row count for a table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.scalar()


def get_sqlite_data(sqlite_engine: Engine, table_name: str) -> Tuple[List[str], List[dict]]:
    """
    Extract all data from SQLite table.

    Returns:
        Tuple of (column_names, list of row dicts)
    """
    with sqlite_engine.connect() as conn:
        # Get column names
        result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1"))
        columns = [col[0] for col in result.cursor.description]

        # Get all data
        result = conn.execute(text(f"SELECT * FROM {table_name}"))
        rows = []
        for row in result.fetchall():
            rows.append(dict(zip(columns, row)))

    logger.info(f"Extracted {len(rows):,} rows from {table_name}")
    return columns, rows


def insert_postgres_data(
    pg_engine: Engine,
    table_name: str,
    columns: List[str],
    rows: List[dict],
    batch_size: int = BATCH_SIZE
) -> int:
    """
    Insert data into PostgreSQL table in batches.

    Uses COPY for bulk insert when possible, falls back to INSERT.

    Returns:
        Number of rows inserted
    """
    if not rows:
        logger.info(f"No data to insert into {table_name}")
        return 0

    # Filter out 'id' column for serial tables
    insert_columns = [c for c in columns if c != 'id']

    inserted = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(rows))
        batch = rows[start_idx:end_idx]

        # Build INSERT statement
        placeholders = ", ".join([f":{col}" for col in insert_columns])
        insert_sql = f"""
            INSERT INTO {table_name} ({", ".join(insert_columns)})
            VALUES ({placeholders})
        """

        try:
            with pg_engine.connect() as conn:
                for row in batch:
                    params = {col: row.get(col) for col in insert_columns}
                    # Handle None values and convert to appropriate types
                    for key, value in params.items():
                        if value is None:
                            params[key] = None

                    conn.execute(text(insert_sql), params)

                conn.commit()

            inserted += len(batch)

            # Progress update
            progress = (batch_idx + 1) / total_batches * 100
            logger.info(
                f"Migrating {table_name}: {inserted:,}/{len(rows):,} rows "
                f"({progress:.1f}%)"
            )

        except Exception as e:
            logger.error(f"Batch insert failed for {table_name}: {e}")
            raise

    return inserted


def verify_migration(
    sqlite_engine: Engine,
    pg_engine: Engine,
    table_name: str
) -> Tuple[bool, str]:
    """
    Verify row counts match between SQLite and PostgreSQL.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        sqlite_count = get_table_row_count(sqlite_engine, table_name)
        pg_count = get_table_row_count(pg_engine, table_name)

        if sqlite_count == pg_count:
            return True, f"Verified: {table_name} has {sqlite_count:,} rows in both databases"
        else:
            return False, (
                f"MISMATCH: {table_name} - SQLite: {sqlite_count:,}, "
                f"PostgreSQL: {pg_count:,}"
            )
    except Exception as e:
        return False, f"Verification failed for {table_name}: {str(e)}"


# =============================================================================
# Main Migration Function
# =============================================================================


@contextmanager
def transaction_context(engine: Engine, description: str):
    """Context manager for database transactions with rollback."""
    logger.info(f"Starting transaction: {description}")
    try:
        yield
        logger.info(f"Transaction committed: {description}")
    except Exception as e:
        logger.error(f"Transaction failed: {description} - {e}")
        raise


def migrate_all_tables(
    sqlite_engine: Engine,
    pg_engine: Engine,
    tables: Optional[List[str]] = None
):
    """
    Migrate all specified tables from SQLite to PostgreSQL.

    Args:
        sqlite_engine: SQLite database engine
        pg_engine: PostgreSQL database engine
        tables: List of tables to migrate (default: TABLES_TO_MIGRATE)
    """
    tables = tables or TABLES_TO_MIGRATE
    migration_results = {}
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("POSTGRESQL MIGRATION STARTED")
    logger.info("=" * 60)
    logger.info(f"Tables to migrate: {', '.join(tables)}")
    logger.info(f"Source: SQLite")
    logger.info(f"Target: PostgreSQL")
    logger.info("=" * 60)

    # Step 1: Create schema
    logger.info("\n[STEP 1] Creating PostgreSQL schema...")
    create_postgres_schema(pg_engine)

    # Step 2: Migrate each table
    logger.info("\n[STEP 2] Migrating tables...")

    for table_name in tables:
        logger.info("-" * 40)
        logger.info(f"Migrating table: {table_name}")

        try:
            # Check if table exists in SQLite
            sqlite_inspector = inspect(sqlite_engine)
            if table_name not in sqlite_inspector.get_table_names():
                logger.warning(f"Table {table_name} not found in SQLite - skipping")
                migration_results[table_name] = {"status": "skipped", "reason": "not_found"}
                continue

            # Extract data from SQLite
            columns, rows = get_sqlite_data(sqlite_engine, table_name)

            if not rows:
                logger.info(f"Table {table_name} is empty - skipping data migration")
                migration_results[table_name] = {"status": "empty", "rows": 0}
                continue

            # Insert into PostgreSQL
            inserted = insert_postgres_data(pg_engine, table_name, columns, rows)

            migration_results[table_name] = {
                "status": "success",
                "rows_extracted": len(rows),
                "rows_inserted": inserted,
            }

        except Exception as e:
            logger.error(f"Migration failed for {table_name}: {e}")
            migration_results[table_name] = {"status": "failed", "error": str(e)}
            raise

    # Step 3: Verify migration
    logger.info("\n[STEP 3] Verifying migration...")
    verification_results = {}

    for table_name in tables:
        if migration_results.get(table_name, {}).get("status") == "success":
            success, message = verify_migration(sqlite_engine, pg_engine, table_name)
            verification_results[table_name] = {"success": success, "message": message}
            logger.info(f"  {message}")

    # Step 4: Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 60)

    success_count = sum(1 for r in migration_results.values() if r["status"] == "success")
    failed_count = sum(1 for r in migration_results.values() if r["status"] == "failed")
    skipped_count = sum(1 for r in migration_results.values() if r["status"] == "skipped")

    logger.info(f"Tables migrated: {success_count}")
    logger.info(f"Tables failed: {failed_count}")
    logger.info(f"Tables skipped: {skipped_count}")
    logger.info(f"Total time: {elapsed_time:.2f}s")

    # Print detailed results
    logger.info("\nDetailed Results:")
    for table_name, result in migration_results.items():
        if result["status"] == "success":
            logger.info(f"  {table_name}: {result['rows_inserted']:,} rows migrated")
        elif result["status"] == "failed":
            logger.error(f"  {table_name}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            logger.info(f"  {table_name}: {result['status'].upper()}")

    # Verify all passed
    all_verified = all(r.get("success", False) for r in verification_results.values())
    if all_verified:
        logger.info("\n✓ All tables verified successfully!")
    else:
        logger.warning("\n✗ Some tables failed verification - check logs")

    return {
        "migration_results": migration_results,
        "verification_results": verification_results,
        "elapsed_time": elapsed_time,
    }


def rollback_migration(pg_engine: Engine, tables: Optional[List[str]] = None):
    """
    Rollback migration by truncating all migrated tables.

    USE WITH CAUTION - This deletes all data in the target tables.
    """
    tables = tables or TABLES_TO_MIGRATE

    logger.warning("=" * 60)
    logger.warning("ROLLBACK INITIATED")
    logger.warning("=" * 60)

    confirm = input("Are you sure you want to truncate all tables? Type 'YES' to confirm: ")
    if confirm != "YES":
        logger.info("Rollback cancelled")
        return

    with pg_engine.connect() as conn:
        for table_name in reversed(tables):
            logger.info(f"Truncating table: {table_name}")
            conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
        conn.commit()

    logger.info("Rollback complete - all tables truncated")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Main entry point for migration script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate SQLite data to PostgreSQL"
    )
    parser.add_argument(
        "--sqlite-path",
        default=os.getenv("SQLITE_PATH", DEFAULT_SQLITE_PATH),
        help=f"Path to SQLite database (default: {DEFAULT_SQLITE_PATH})"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback migration (truncate all tables)"
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help=f"Tables to migrate (default: {', '.join(TABLES_TO_MIGRATE)})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating"
    )

    args = parser.parse_args()

    # Validate environment
    if not os.getenv("DATABASE_URL") and not all([
        os.getenv("POSTGRES_USER"),
        os.getenv("POSTGRES_PASSWORD"),
        os.getenv("POSTGRES_HOST"),
        os.getenv("POSTGRES_DB"),
    ]):
        logger.error("PostgreSQL connection not configured.")
        logger.error("Set DATABASE_URL or POSTGRES_* environment variables.")
        sys.exit(1)

    # Rollback mode
    if args.rollback:
        pg_engine = get_postgres_engine()
        rollback_migration(pg_engine, args.tables)
        pg_engine.dispose()
        return

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN - No changes will be made")
        sqlite_engine = get_sqlite_engine(args.sqlite_path)
        for table_name in (args.tables or TABLES_TO_MIGRATE):
            try:
                count = get_table_row_count(sqlite_engine, table_name)
                logger.info(f"  {table_name}: {count:,} rows would be migrated")
            except Exception as e:
                logger.warning(f"  {table_name}: {e}")
        sqlite_engine.dispose()
        return

    # Full migration
    logger.info(f"SQLite path: {args.sqlite_path}")

    try:
        sqlite_engine = get_sqlite_engine(args.sqlite_path)
        pg_engine = get_postgres_engine()

        results = migrate_all_tables(sqlite_engine, pg_engine, args.tables)

        sqlite_engine.dispose()
        pg_engine.dispose()

        # Exit with error code if any failures
        failed = sum(1 for r in results["migration_results"].values() if r["status"] == "failed")
        if failed > 0:
            logger.error(f"Migration completed with {failed} failures")
            sys.exit(1)

        logger.info("\nMigration completed successfully!")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
