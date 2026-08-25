import os
import re

dirs = ['backend', 'ml', 'tests']
patterns = {
    r'from ml\.predict': 'from ml.inference.predict',
    r'import ml\.predict': 'import ml.inference.predict',
    r'from ml\.ensemble': 'from ml.inference.ensemble',
    r'import ml\.ensemble': 'import ml.inference.ensemble',
    r'from ml\.features': 'from ml.utils.features',
    r'import ml\.features': 'import ml.utils.features',
    r'from ml\.drift': 'from ml.utils.drift',
    r'import ml\.drift': 'import ml.utils.drift',
    r'from ml\.train': 'from ml.training.train',
    r'import ml\.train': 'import ml.training.train',
    r'from ml\.retrain': 'from ml.training.retrain',
    r'import ml\.retrain': 'import ml.training.retrain',
    r'from ml\.evaluate': 'from ml.training.evaluate',
    r'import ml\.evaluate': 'import ml.training.evaluate',
    r'from ml import predict': 'from ml.inference import predict',
    r'from ml import features': 'from ml.utils import features',
    r'from ml import drift': 'from ml.utils import drift',
    r'from ml import train': 'from ml.training import train',
    r'from ml import retrain': 'from ml.training import retrain',
    r'from ml import evaluate': 'from ml.training import evaluate',
    r'from ml import ensemble': 'from ml.inference import ensemble',
}

files_changed = 0
for d in dirs:
    if not os.path.exists(d): continue
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                new_content = content
                for k, v in patterns.items():
                    new_content = re.sub(k, v, new_content)
                if new_content != content:
                    with open(path, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    files_changed += 1
                    print(f"Updated {path}")

print(f'Updated {files_changed} files.')
