import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_and_prepare_data(csv_path):
    """Loads and prepares the bank marketing data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at: {csv_path}")

    data = pd.read_csv(csv_path, sep=";")

    # Rename columns
    data.rename(
        columns={
            "marital": "marital_status",
            "default": "credit_default",
            "housing": "housing_loan",
            "loan": "personal_loan",
            "y": "response",
        },
        inplace=True,
    )

    # Drop poutcome column
    if "poutcome" in data.columns:
        data.drop("poutcome", axis=1, inplace=True)

    return data


def get_preprocessing_pipeline():
    """Returns the preprocessing pipeline."""
    numeric_features = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]
    categorical_features = [
        "job",
        "marital_status",
        "education",
        "month",
        "housing_loan",
        "personal_loan",
        "contact",
        "credit_default",
    ]

    pre_processor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", pre_processor)])


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Trains and evaluates multiple classification models."""
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100
        ),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }

    results = {}
    trained_models = {}

    with tqdm(models.items()) as pbar:
        for name, model in pbar:
            pbar.set_description(f"Training {name}")
            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()
            training_time = end_time - start_time

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # print(f"\nAccuracy for {name}: {accuracy:.4f}")
            # print(f"ROC AUC Score for {name}: {roc_auc:.4f}")
            # print(f"Training Time for {name}: {training_time:.2f}s")

            results[name] = {
                "Accuracy": accuracy,
                "ROC_AUC_Score": roc_auc,
                "Training Time (s)": training_time,
            }
            trained_models[name] = model

    return trained_models, pd.DataFrame.from_dict(
        results, orient="index"
    ).sort_values(by="ROC_AUC_Score", ascending=False)


def save_artifacts(models, pipeline, encoder, save_dir):
    """Saves trained models and preprocessing artifacts."""
    save_dir.mkdir(exist_ok=True)

    name_map = {
        "Logistic Regression": "logreg",
        "Decision Tree": "dtree",
        "Random Forest": "rforest",
        "Gradient Boosting": "gbm",
        "Gaussian Naive Bayes": "gnb",
        "K-Nearest Neighbors": "knn",
    }

    for name, model in models.items():
        short_name = name_map.get(name, name.lower().replace(" ", "_"))
        filename = short_name + "_model.pkl"
        joblib.dump(model, save_dir / filename)

    joblib.dump(pipeline, save_dir / "pre_pipeline.pkl")
    joblib.dump(encoder, save_dir / "label_encoder.pkl")
    print(f"\nModels and artifacts saved to '{save_dir}'")


def main():
    """Main function to run the full pipeline."""
    # Define paths relative to the project root
    csv_path = Path("dataset") / "bank.csv"
    save_dir = Path("saved_models")

    # Load and prepare data
    data = load_and_prepare_data(csv_path)

    # Define features and target
    X = data.drop(columns=["response"])
    y = data["response"]

    # Split data
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=78
    )

    # Preprocess target variable
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    # Preprocess features
    pre_pipeline = get_preprocessing_pipeline()
    X_train_processed = pre_pipeline.fit_transform(X_train)
    X_test_processed = pre_pipeline.transform(X_test)

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=78)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed, y_train
    )

    # Train and evaluate models
    trained_models, results_df = train_and_evaluate_models(
        X_train_resampled, y_train_resampled, X_test_processed, y_test
    )

    print("\n--- Model Performance Summary ---")
    print(results_df)

    # Save models and artifacts
    save_artifacts(trained_models, pre_pipeline, label_encoder, save_dir)


if __name__ == "__main__":
    main()

