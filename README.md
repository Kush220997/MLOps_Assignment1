# MLOps Assignment 1

This repository implements a complete MLOps workflow for predicting house prices using the Boston Housing dataset and classic scikit-learn regression models.

## Setup and Installation

Below steps are followed to set up local environment and install required dependencies.

1. Repository setup
```bash
# Clone the repository
git clone https://github.com/Kush220997/MLOps_Assignment1
cd MLOps_Assignment1
```

2. Environment activation
```bash
# Create and activate a dedicated Conda environment with Python 3.10
conda create -n mlops_env python=3.10 -y
conda activate mlops_env
```

3. Install dependencies
The required packages are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

## How to Run

The main pipeline functions are implemented as generic functions in `misc.py` (data loading, preprocessing, training, testing).

Training individual models
Run model-specific scripts to train and print performance.

- Decision Tree Regressor
```bash
python train.py
# Branch origin: dtree (merged to main)
```

- Kernel Ridge
```bash
python train2.py
# Branch origin: kernelridge
```

## MLOps Automation

A CI workflow is configured to validate code and model performance.

- Workflow file: `.github/workflows/ci.yml`
- Trigger: any push event to the `kernelridge` branch
- Pipeline steps:
    1. Checkout code
    2. Install dependencies from `requirements.txt`
    3. Run Decision Tree training script
    4. Run Kernel Ridge training script

## key files
- README.md
- misc.py               — shared pipeline utilities (load, preprocess, train, test)
- train.py              — trains DecisionTreeRegressor
- train2.py             — trains KernelRidge
- requirements.txt
- .github/workflows/ci.yml