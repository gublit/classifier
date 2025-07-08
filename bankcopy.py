import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


class BankingClassifier:
    def __init__(self, data_path, models_dir, plots_dir):
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.plots_dir = Path(plots_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.data = None
        self.pre_pipeline = None
        self.label_encoder = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.models = {
            "Logistic Regression": LogisticRegression(class_weight='balanced'),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }
        self.results = pd.DataFrame(columns=["Model", "Accuracy", "ROC AUC Score"])

    def load_and_prepare_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        self.data = pd.read_csv(self.data_path, sep=';')
        print("Data loaded successfully.")
        print(self.data.head())
        self._rename_and_change_types()
        if 'poutcome' in self.data.columns:
            self.data.drop("poutcome", axis=1, inplace=True)

    def _rename_and_change_types(self):
        self.data.rename(
            columns={
                "marital": "marital_status",
                "default": "credit_default",
                "housing": "housing_loan",
                "loan": "personal_loan",
                "y": "response",
            },
            inplace=True,
        )
        for col in [
            "response", "marital_status", "education", "job", "contact",
            "month", "day", "credit_default", "housing_loan", "personal_loan",
        ]:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype("category")
        print("Columns renamed and data types changed.")

    def perform_eda(self):
        print("Performing Exploratory Data Analysis...")
        # Distribution of response variable
        plt.figure(figsize=(8, 6))
        sns.countplot(x="response", data=self.data)
        plt.title("Distribution of Response Variable")
        plt.savefig(self.plots_dir / "response_distribution.png")
        plt.close()

        # Correlation matrix for numeric features
        numeric_ft = self.data.select_dtypes(include=["number"]).columns
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.data[numeric_ft].corr(), annot=True, cmap="coolwarm", fmt=".2f"
        )
        plt.title("Correlation Matrix of Numeric Features")
        plt.savefig(self.plots_dir / "correlation_matrix.png")
        plt.close()
        print(f"EDA plots saved to {self.plots_dir}")

    def preprocess_and_balance_data(self):
        X = self.data.drop(columns=["response"])
        y = self.data["response"]

        numeric_features = X.select_dtypes(include=["number"]).columns
        categorical_features = X.select_dtypes(include=["category", "object"]).columns

        pre_processor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ]
        )

        self.pre_pipeline = Pipeline(steps=[("preprocessor", pre_processor)])
        
        X_train_orig, self.X_test, y_train_orig, self.y_test = train_test_split(
            X, y, train_size=0.8, stratify=y, random_state=78
        )

        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train_orig)
        self.y_test = self.label_encoder.transform(self.y_test)

        X_train_processed = self.pre_pipeline.fit_transform(X_train_orig)
        self.X_test = self.pre_pipeline.transform(self.X_test)
        
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=78)
        self.X_train, self.y_train = smote.fit_resample(X_train_processed, y_train_encoded)

        print("Data preprocessing and balancing complete.")
        print(f"Original X_train shape: {X_train_processed.shape}")
        print(f"Resampled X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")

    def train_and_evaluate_models(self):
        plt.figure(figsize=(10, 8))

        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            print(f"\n--- {name} ---")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC AUC Score: {roc_auc:.4f}")
            print("Classification Report:\n", classification_report(self.y_test, y_pred))

            new_row = pd.DataFrame([{"Model": name, "Accuracy": accuracy, "ROC AUC Score": roc_auc}])
            self.results = pd.concat([self.results, new_row], ignore_index=True)

            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve Comparison")
        plt.legend(loc="lower right")
        plt.grid(True)
        roc_plot_path = self.plots_dir / "roc_comparison.png"
        plt.savefig(roc_plot_path)
        plt.close()

        self.results = self.results.sort_values(by="ROC AUC Score", ascending=False)
        print("\n--- Model Comparison ---")
        print(self.results)

    def save_artifacts(self):
        print("Saving models and preprocessing artifacts...")
        for name, model in self.models.items():
            model_path = self.models_dir / f"{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_path)

        joblib.dump(self.pre_pipeline, self.models_dir / "pre_pipeline.pkl")
        joblib.dump(self.label_encoder, self.models_dir / "label_encoder.pkl")
        print(f"Artifacts saved to {self.models_dir}")

    def predict_on_new_data(self, new_data):
        if self.pre_pipeline is None:
            raise Exception("Preprocessing pipeline is not fitted. Run preprocessing first.")

        if not isinstance(new_data, pd.DataFrame):
            raise TypeError("new_data must be a pandas DataFrame.")

        print("\nMaking prediction on new data...")
        
        # Ensure categorical columns have the correct type for the pipeline
        for col in new_data.select_dtypes(include=["object"]).columns:
            new_data[col] = new_data[col].astype("category")
            
        processed_new_data = self.pre_pipeline.transform(new_data);

        best_model_name = self.results.iloc[0]["Model"]
        best_model = self.models[best_model_name]

        prediction_encoded = best_model.predict(processed_new_data)
        prediction_proba = best_model.predict_proba(processed_new_data)

        prediction = self.label_encoder.inverse_transform(prediction_encoded)

        print(f"Best Model ({best_model_name}) Prediction: {prediction[0]}")
        print(f"Prediction Probabilities: {prediction_proba[0]}")
        return prediction, prediction_proba

    def run(self):
        self.load_and_prepare_data()
        self.perform_eda()
        self.preprocess_and_balance_data()
        self.train_and_evaluate_models()
        self.save_artifacts()

        new_sample = pd.DataFrame({
            "age": [30], "balance": [1000], "day": [15], "duration": [200],
            "campaign": [1], "pdays": [999], "previous": [0], "job": ["admin."],
            "marital_status": ["single"], "education": ["university.degree"],
            "contact": ["cellular"], "month": ["may"], "housing_loan": ["yes"],
            "personal_loan": ["no"], "credit_default": ["no"],
        })
        
        self.predict_on_new_data(new_sample)


if __name__ == "__main__":
    classifier = BankingClassifier(
        data_path="dataset/bank.csv",
        models_dir="saved_models",
        plots_dir="plots",
    )
    classifier.run()
