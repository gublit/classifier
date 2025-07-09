# Bank Marketing Classification Project

## Overview
This project demonstrates a comprehensive workflow for building, evaluating, and deploying machine learning models for tabular classification tasks using the Bank Marketing dataset from the UCI Machine Learning Repository. The primary goal is to predict whether a customer will subscribe to a term deposit based on their demographic and behavioral features.

## Project Structure
```
├── bank.py
├── bankcopy.py
├── banking.py
├── environment.yaml
├── LICENSE
├── README.md
├── requirements.txt
├── roc_comparison.png
├── support_vector_model.pkl
├───.git/
├───.vscode/
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
- **bank.py**: A basic script that loads data, preprocesses it, trains multiple models, evaluates them, and saves the models and preprocessing pipeline.
- **banking.py**: An advanced version of `bank.py` with logging, hyperparameter tuning for Logistic Regression, and plotting of a consolidated ROC curve.
- **bankcopy.py**: A class-based implementation of the same workflow. It encapsulates the logic into a `BankingClassifier` class, which makes it more modular and reusable. It also includes EDA plot generation.

### Data
- **dataset/bank.csv**: The main dataset used for training and evaluation.

### Models
- **saved_models/** and **dataset/saved_models/**: Serialized model files (e.g., `.pkl` files for scikit-learn models, label encoders, and preprocessing pipelines).

### Plots
- **plots/**: Visualizations generated during EDA and model evaluation (e.g., correlation matrix, ROC curves).

### Environment & Dependencies
- **requirements.txt**: Python package dependencies for the project.
- **environment.yaml**: Conda environment configuration file.

## How to Run
1. **Set up the environment**
   - Using Conda:
     ```sh
     conda env create -f environment.yaml
     conda activate clasfi
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
     or
      ```sh
     python banking.py
     ```
     or
      ```sh
     python bankcopy.py
     ```

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
- pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib, (see requirements.txt or environment.yaml for details)

## License
See [LICENSE](LICENSE) for details.

## References
- [UCI Machine Learning Repository: Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- scikit-learn documentation: https://scikit-learn.org/
- imbalanced-learn documentation: https://imbalanced-learn.org/

---
For questions or contributions, please open an issue or pull request.
