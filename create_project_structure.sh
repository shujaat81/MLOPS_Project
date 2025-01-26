#!/bin/bash

# Create main project directory
mkdir -p project

# Create all directories
mkdir -p project/data/{raw,processed,versioned}
mkdir -p project/models/{trained,tuned}
mkdir -p project/notebooks
mkdir -p project/src/utils
mkdir -p project/deployment/{helm/templates,k8s}
mkdir -p project/ci_cd/{github,gitlab,scripts,logs}
mkdir -p project/tracking/mlruns
mkdir -p project/tests/{unit,integration}

# Create empty files
touch project/data/dvc.yaml
touch project/models/model_metadata.json
touch project/notebooks/{experimentation.ipynb,data_versioning.ipynb}
touch project/src/{preprocess.py,train.py,tune.py,evaluate.py,serve.py}
touch project/deployment/{Dockerfile,requirements.txt}
touch project/deployment/helm/values.yaml
touch project/ci_cd/github/main.yml
touch project/ci_cd/gitlab/.gitlab-ci.yml
touch project/tracking/config.yaml
touch project/tests/unit/{test_preprocess.py,test_train.py,test_serve.py}
touch project/{.gitignore,README.md,LICENSE,summary_report.md}

# Add basic content to .gitignore
cat << 'EOF' > project/.gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/

# MLflow
mlruns/

# DVC
/data/versioned
*.dvc

# Model files
*.pkl
*.h5
*.joblib

# Logs
*.log
EOF

# Add basic content to README.md
cat << 'EOF' > project/README.md
# Machine Learning Project

## Project Structure
This project follows a structured approach for machine learning development, deployment, and monitoring.

### Directory Structure
- `data/`: Dataset storage and versioning
- `models/`: Trained and tuned models
- `notebooks/`: Jupyter notebooks for experimentation
- `src/`: Source code for the project
- `deployment/`: Deployment configurations
- `ci_cd/`: CI/CD pipeline configurations
- `tracking/`: MLflow experiment tracking
- `tests/`: Unit and integration tests

## Setup
[Instructions for setting up the project]

## Usage
[Instructions for using the project]

## License
[License information]
EOF

echo "Project structure created successfully!" 