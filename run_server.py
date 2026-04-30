"""
Entry point for the Mortgage AI backend.
Runs api.py as the ASGI app on port 8001.

Usage:
    python run_server.py
    OR (for reload mode):
    uvicorn run_server:app --host 0.0.0.0 --port 8001 --reload
"""

import importlib.util
import sys
import os

# Load api.py explicitly (avoids conflict with api/ package)
_spec = importlib.util.spec_from_file_location(
    "mortgage_api",
    os.path.join(os.path.dirname(__file__), "api.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["mortgage_api"] = _mod
_spec.loader.exec_module(_mod)

# Expose app for uvicorn
app = _mod.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "run_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_excludes=["mortgage-frontend/*", "*.pkl", "*.joblib", "*.csv"]
    )
