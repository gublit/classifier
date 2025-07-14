# Bank Marketing Classification Project

## Overview
This project provides a complete workflow for building and evaluating machine learning models for a binary classification task. It uses the Bank Marketing dataset from the UCI Machine Learning Repository to predict whether a customer will subscribe to a term deposit.

The project is presented in two formats:
- A modular, object-oriented Python script (`main.py`) for a streamlined, automated pipeline.
- An interactive Jupyter Notebook (`notebooks/main.ipynb`) for detailed, step-by-step exploration and visualization.

## Project Structure
```
├── main.py
├── notebooks/
│   └── main.ipynb
├── environment.yaml
├── dataset/
│   └── bank.csv
├── plots/
│   └── (contains output plots)
└── saved_models/
    └── (contains saved model artifacts)
```

## Key Components

### Main Script (`main.py`)
- The primary script for the entire automated workflow. It handles:
  - Data loading and preprocessing.
  - Exploratory Data Analysis (EDA) and plot generation.
  - Training and evaluating 8 different classification models.
  - Handling class imbalance using SMOTE.
  - Saving all artifacts (models, plots, encoders).
  - Predicting on a sample data point using the top 3 best-performing models.

### Jupyter Notebook (`notebooks/main.ipynb`)
- An interactive notebook that provides a detailed, cell-by-cell walkthrough of the entire process:
  - In-depth data exploration and visualization.
  - Clear explanations for each preprocessing and modeling step.
  - Training and evaluation of the same 8 models.
  - Visualization of ROC curves and confusion matrices for all models.
  - A great tool for understanding the project's methodology.

### Data
- **dataset/bank.csv**: The raw data used for training and evaluation.

### Artifacts
- **saved_models/**: The output directory for all generated artifacts, including trained models (`.pkl`), the preprocessing pipeline, and the label encoder.
- **plots/**: The output directory for all visualizations, such as the response distribution, correlation matrix, and ROC curve comparisons.

### Environment
- **environment.yaml**: The Conda environment file, which contains all the necessary dependencies to ensure a reproducible setup.

## How to Run

### 1. Set up the Conda Environment
```sh
conda env create -f environment.yaml
conda activate clasfi
```

### 2. Run the Main Script
To execute the entire automated pipeline:
```sh
python main.py
```
This will run the complete workflow, save all artifacts, and print the model performance summary and sample predictions to the console.

### 3. Use the Jupyter Notebook
For an interactive experience:
```sh
jupyter notebook notebooks/main.ipynb
```

## Features
- End-to-end classification pipeline in both a script and a notebook.
- Object-oriented design in `main.py` for modularity and reusability.
- Integrated EDA and visualization.
- Comparison of 8 models: `Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`, `Gaussian Naive Bayes`, `K-Nearest Neighbors`, `XGBoost`, and `LightGBM`.
- Class imbalance handled with SMOTE.
- Detailed progress bars using `rich` to monitor model training in the script.

## Requirements
- Python 3.11
- See `environment.yaml` for a full list of dependencies.

## License
See [LICENSE](LICENSE) for details.

## References
- [UCI Machine Learning Repository: Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- scikit-learn documentation: https://scikit-learn.org/
- imbalanced-learn documentation: https://imbalanced-learn.org/
---
For questions or contributions, please open an issue or pull request.