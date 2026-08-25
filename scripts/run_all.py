import os
import subprocess
import sys
import time

def run_all():
    print("Starting Mortgage AI Project...")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set PYTHONPATH to project root
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir
    
    # Start Backend
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.run_server:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
    print("Starting Backend...")
    backend_proc = subprocess.Popen(backend_cmd, env=env, cwd=root_dir)
    
    time.sleep(3)
    
    # Start Frontend
    frontend_dir = os.path.join(root_dir, "frontend")
    frontend_cmd = ["npm", "run", "dev"]
    print("Starting Frontend...")
    
    # Use shell=True for npm on Windows
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=frontend_dir, shell=True)
    
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == '__main__':
    run_all()
