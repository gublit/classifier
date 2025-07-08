import logging
from pathlib import Path
import warnings

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data(csv_path):
    """Loads and prepares the bank marketing data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at: {csv_path}")
    
    data = pd.read_csv(csv_path, sep=';')
    
    # Drop poutcome column as it has too many unknown values
    if 'poutcome' in data.columns:
        data.drop('poutcome', axis=1, inplace=True)
        
    return data

def get_preprocessing_pipeline():
    """Returns the preprocessing pipeline."""
    numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month']
    
    pre_processor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop' 
    )
    return Pipeline(steps=[('preprocessor', pre_processor)])

def train_and_evaluate(models, X_train, y_train, X_test, y_test, save_dir):
    """Trains, evaluates, and saves multiple models."""
    y_pred_proba_results = {}
    for model_name, model in tqdm(models, desc="Training Models"):
        # logging.info(f"--- Training {model_name} ---")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        # logging.info(f"Accuracy: {accuracy:.4f}")
        # logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
        
        # Sanitize filename
        sanitized_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_filename = f'{sanitized_name}_model.pkl'
        model_path = save_dir / model_filename
        joblib.dump(model, model_path)
        # logging.info(f"Model saved to {model_path}")
        
        y_pred_proba_results[model_name] = y_pred_proba
        
    return y_pred_proba_results

def plot_roc_curves(y_test, y_pred_proba_dict, save_path):
    """Plots and saves a consolidated ROC curve chart."""
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
    plt.savefig(save_path)
    plt.close()
    logging.info(f"ROC comparison plot saved to {save_path}")

def main():
    """Main execution function."""
    # Define paths
    base_path = Path(__file__).parent
    data_path = base_path / 'dataset' / 'bank.csv'
    artifacts_dir = base_path / 'saved_models'
    artifacts_dir.mkdir(exist_ok=True)

    # Load data
    data = load_and_prepare_data(data_path)

    # Define features and target
    X = data.drop(columns=['y'])
    y = data['y']

    # Split data
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, train_size=0.8, stratify=y, random_state=78)

    # Encode target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_raw)
    y_test_encoded = label_encoder.transform(y_test_raw)
    joblib.dump(label_encoder, artifacts_dir / 'label_encoder.pkl')

    # Preprocess features
    pre_pipeline = get_preprocessing_pipeline()
    X_train_processed = pre_pipeline.fit_transform(X_train)
    X_test_processed = pre_pipeline.transform(X_test)
    joblib.dump(pre_pipeline, artifacts_dir / 'pre_pipeline.pkl')

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=78)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train_encoded)

    # Define models
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier(n_estimators=100)),
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('Gaussian Naive Bayes', GaussianNB()),
        ('K Neighbors', KNeighborsClassifier())
    ]

    # Hyperparameter Tuning for Logistic Regression
    logging.info("Performing hyperparameter tuning for Logistic Regression...")
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    logging.info(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
    tuned_log_reg = grid_search.best_estimator_
    models[0] = ('Logistic Regression (Tuned)', tuned_log_reg)

    # Train and evaluate all models
    y_pred_probas = train_and_evaluate(models, X_train_resampled, y_train_resampled, X_test_processed, y_test_encoded, artifacts_dir)

    # Plot ROC curves
    plot_roc_curves(y_test_encoded, y_pred_probas, artifacts_dir / 'roc_comparison.png')

    logging.info("Script finished successfully.")

if __name__ == '__main__':
    main()
