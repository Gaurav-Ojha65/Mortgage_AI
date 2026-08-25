import os
import shutil

def clean():
    print("Cleaning project...")
    # Clean __pycache__
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                shutil.rmtree(os.path.join(root, d))
    
    # Clean temp files
    for root, _, files in os.walk('.'):
        for f in files:
            if f.endswith(('.log', '.tmp', '.DS_Store')):
                os.remove(os.path.join(root, f))
                
    print("Project cleaned!")

if __name__ == '__main__':
    clean()
