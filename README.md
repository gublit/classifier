# Bank Marketing Classification Project

## Overview
This project provides a complete workflow for building and evaluating machine learning models for a binary classification task. It uses the Bank Marketing dataset from the UCI Machine Learning Repository to predict whether a customer will subscribe to a term deposit.

## Project Structure
```
├── main.py
├── environment.yaml
├── LICENSE
├── README.md
├── dataset/
│   └── bank.csv
├── notebooks/
│   └── (contains notebooks for exploration)
├── plots/
│   └── (contains output plots)
└── saved_models/
    └── (contains saved model artifacts)
```

## Key Components

### Main Script
- **main.py**: The primary script for the entire workflow. It handles:
  - Data loading and preprocessing.
  - Exploratory Data Analysis (EDA).
  - Training and evaluating multiple classification models.
  - Hyperparameter tuning for Logistic Regression.
  - Saving all artifacts (models, plots, and encoders).

### Data
- **dataset/bank.csv**: The raw data used for training and evaluation.

### Artifacts
- **saved_models/**: The output directory for all generated artifacts, including trained models (`.pkl`), the preprocessing pipeline, and the label encoder.
- **plots/**: The output directory for all visualizations, such as the response distribution and ROC curve comparisons.

### Environment
- **environment.yaml**: The Conda environment file, which contains all the necessary dependencies to ensure a reproducible setup.

## How to Run

1.  **Set up the Conda Environment**:
    ```sh
    conda env create -f environment.yaml
    conda activate clasfi
    ```

2.  **Run the Main Script**:
    ```sh
    python main.py
    ```
    This will execute the entire pipeline, from data loading to model saving, and will print the progress to the console. The script will also output predictions from the top 3 best performing models on a sample data point.

## Features
- End-to-end classification pipeline in a single, well-structured script.
- Object-oriented design for modularity and reusability.
- Integrated EDA and visualization.
- Compares multiple models and uses the top 3 for prediction.
- Detailed progress bars to monitor model training.

## Requirements
- Python 3.11 (as specified in `environment.yaml`)
- See `environment.yaml` for a full list of dependencies.

## License
See [LICENSE](LICENSE) for details.

## References
- [UCI Machine Learning Repository: Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- scikit-learn documentation: https://scikit-learn.org/
- imbalanced-learn documentation: https://imbalanced-learn.org/
---
For questions or contributions, please open an issue or pull request.