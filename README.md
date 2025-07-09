# Bank Marketing Classification Project

## Overview
This project demonstrates a comprehensive workflow for building, evaluating, and deploying machine learning models for tabular classification tasks using the Bank Marketing dataset from the UCI Machine Learning Repository. The primary goal is to predict whether a customer will subscribe to a term deposit based on their demographic and behavioral features.

## Project Structure
```
├── bank.py
├── bankcopy.py
├── banking.py
├── env.yml / environment.yaml
├── LICENSE
├── README.md
├── requirements.txt
├── roc_comparison.png
├── support_vector_model.pkl
├── teams_data.csv
├── dataset/
│   ├── bank.csv
│   ├── decision_tree_model.pkl
│   ├── gaussian_naive_bayes_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── k_neighbors_model.pkl
│   ├── label_encoder.pkl
│   ├── logistic_regression_(tuned)_model.pkl
│   ├── pre_pipeline.pkl
│   ├── random_forest_model.pkl
│   ├── roc_comparison.png
│   └── saved_models/
│       └── ... (model files)
├── notebooks/
│   ├── bank_model.pkl
│   ├── bank.ipynb
│   ├── banking.ipynb
│   ├── tempo.ipynb
│   └── test.ipynb
├── plots/
│   ├── correlation_matrix.png
│   ├── response_distribution.png
│   ├── roc_comparison.png
│   └── roc_curves.png
├── saved_models/
│   └── ... (model files)
```

## Key Components

### Notebooks
- **notebooks/bank.ipynb**: Main notebook for data preprocessing, exploratory data analysis (EDA), model training, evaluation, and prediction. Includes code for:
  - Data cleaning and feature engineering
  - Handling class imbalance with SMOTE
  - Training multiple classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Gaussian Naive Bayes, K-Nearest Neighbors)
  - Model evaluation (accuracy, ROC AUC, confusion matrix, classification report)
  - Model comparison and saving
  - Making predictions on new data
- **notebooks/banking.ipynb, tempo.ipynb, test.ipynb**: Additional or experimental notebooks for model development and testing.

### Scripts
- **bank.py, bankcopy.py, banking.py**: Python scripts for data processing, model training, or utility functions. (See code for details.)

### Data
- **dataset/bank.csv**: The main dataset used for training and evaluation.
- **teams_data.csv**: Additional data (purpose may vary).

### Models
- **saved_models/** and **dataset/saved_models/**: Serialized model files (e.g., `.pkl` files for scikit-learn models, label encoders, and preprocessing pipelines).

### Plots
- **plots/**: Visualizations generated during EDA and model evaluation (e.g., correlation matrix, ROC curves).

### Environment & Dependencies
- **requirements.txt**: Python package dependencies for the project.
- **env.yml / environment.yaml**: Conda environment configuration files.

## How to Run
1. **Set up the environment**
   - Using Conda:
     ```sh
     conda env create -f environment.yaml
     conda activate <env_name>
     ```
   - Or using pip:
     ```sh
     pip install -r requirements.txt
     ```
2. **Open and run the main notebook**
   - Launch JupyterLab or VS Code and open `notebooks/bank.ipynb`.
   - Run all cells to reproduce the workflow: data loading, preprocessing, EDA, model training, evaluation, and prediction.

3. **Scripts**
   - You may also run the Python scripts directly for batch processing or automation:
     ```sh
     python bank.py
     ```
   - (See script files for specific usage.)

## Features
- End-to-end tabular classification pipeline
- Data cleaning and feature engineering
- Handling class imbalance with SMOTE
- Multiple model training and comparison
- Model persistence (saving/loading with joblib)
- Visualizations for EDA and model evaluation
- Easy extension for new data or additional models

## Requirements
- Python 3.8+
- pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib, (see requirements.txt)

## License
See [LICENSE](LICENSE) for details.

## References
- [UCI Machine Learning Repository: Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- scikit-learn documentation: https://scikit-learn.org/
- imbalanced-learn documentation: https://imbalanced-learn.org/

---
For questions or contributions, please open an issue or pull request.
