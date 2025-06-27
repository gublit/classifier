# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
data = pd.read_csv('/home/tisinr/Dev/models/classifier/dataset/bank.csv', header=0, sep=';')
print(data.head())

# Assign features and labels (classic/original)
X = data.drop(columns=['y'])
y = data['y']

# Classic feature lists (example, adjust as needed)
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'month', 'housing', 'loan', 'default']

pre_processor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
pre_pipeline = Pipeline(steps=[
    ('preprocessor', pre_processor)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=78)

# Label encode the target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Fit preprocessing on training data and transform both sets
X_train = pre_pipeline.fit_transform(X_train)
X_test = pre_pipeline.transform(X_test)

# Save the preprocessor
joblib.dump(pre_pipeline, 'pre_pipeline.pkl')

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Define models
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('Support Vector', SVC(probability=True)),
    ('Gaussian Naive Bayes', GaussianNB()),
    ('K Neighbors', KNeighborsClassifier())
]

from sklearn.model_selection import GridSearchCV, StratifiedKFold
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    """Trains a model, evaluates it, and prints metrics."""
    logging.info(f"--- Training {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    # Save the trained model
    joblib.dump(model, f'{model_name.lower().replace(" ", "_")}_model.pkl')
    return y_pred_proba

def plot_roc_curve(y_test, y_pred_proba_dict, filename='roc_comparison.png'):
    """Plots ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    for model_name, y_pred_proba in y_pred_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Model Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# --- Main execution flow ---
if __name__ == '__main__':
    logging.info("Starting model training and evaluation.")

    # Hyperparameter Tuning Example (Logistic Regression)
    logging.info("Performing hyperparameter tuning for Logistic Regression...")
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
    logging.info(f"Best AUC score for Logistic Regression: {grid_search.best_score_:.4f}")
    tuned_log_reg = grid_search.best_estimator_
    models[0] = ('Logistic Regression (Tuned)', tuned_log_reg) # Replace original LR with tuned one

    y_pred_proba_results = {}
    for model_name, model in models:
        y_pred_proba = train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
        y_pred_proba_results[model_name] = y_pred_proba

    plot_roc_curve(y_test, y_pred_proba_results)
    logging.info("Model training and evaluation complete. ROC curve saved.")
