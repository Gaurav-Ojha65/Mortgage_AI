import os
import tempfile
import pytest

# Create a temporary SQLite database file for tests
fd, temp_db_path = tempfile.mkstemp(suffix=".db")
os.close(fd)

# Crucial: Override DATABASE_PATH globally BEFORE any backend modules are imported.
# This ensures that `from backend.database import DATABASE_PATH` fetches our temporary path.
os.environ["DATABASE_PATH"] = temp_db_path

@pytest.fixture(scope="session", autouse=True)
def isolated_db():
    """
    Session-scoped autouse fixture that initializes the test schema on the temporary DB
    and cleans it up after the test suite completes.
    """
    from backend.api import init_db
    from backend.auth import init_users_table
    from backend.audit_log import init_audit_table
    from backend.database import DATABASE_PATH
    
    # Assert isolation is working
    assert DATABASE_PATH == temp_db_path
    
    # Initialize schema
    init_db()
    init_users_table()
    init_audit_table()
    
    yield temp_db_path
    
    # Cleanup after test session
    if os.path.exists(temp_db_path):
        try:
            os.remove(temp_db_path)
        except PermissionError:
            pass
