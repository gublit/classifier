import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from rich.progress import track
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


class TD_Predictor:
    """A class to handle the training and prediction of various machine learning models
    on a banking dataset, including data loading, preprocessing, model training,
    evaluation, and saving of artifacts.
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self.base_dir: Path = Path(base_dir)
        self.data_path: Path = self.base_dir / "dataset" / "bank.csv"
        self.models_dir: Path = self.base_dir / "saved_models"
        self.plots_dir: Path = self.base_dir / "plots"
        self.models_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.data: Optional[pd.DataFrame] = None
        self.pre_pipeline: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.X_train: Optional[NDArray[Any]] = None
        self.X_test: Optional[NDArray[Any]] = None
        self.y_train: Optional[NDArray[Any]] = None
        self.y_test: Optional[NDArray[Any]] = None
        self.models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced", max_iter=1000
            ),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced"
            ),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "XGBoost": XGBClassifier(),
            "LightGBM": LGBMClassifier(class_weight="balanced"),
        }
        self.results: pd.DataFrame = pd.DataFrame(
            columns=["Model", "Accuracy", "ROC AUC Score", "Training Time (s)"]
        )

    def load_and_prepare_data(self) -> None:
        print("Loading and preparing data...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        self.data = pd.read_csv(self.data_path, sep=";")
        self._rename_and_change_types()
        if self.data is not None and "poutcome" in self.data.columns:
            self.data.drop("poutcome", axis=1, inplace=True)
        print("Data loaded and prepared successfully.")

    def _rename_and_change_types(self) -> None:
        if self.data is None:
            return
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
            "response",
            "marital_status",
            "education",
            "job",
            "contact",
            "month",
            "day",
            "credit_default",
            "housing_loan",
            "personal_loan",
        ]:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype("category")

    def eda(self) -> None:
        print("Performing Exploratory Data Analysis...")
        if self.data is None:
            print("No data to perform EDA on. Load data first.")
            return
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

    def preprocess_and_balance_data(self) -> None:
        print("Preprocessing and balancing data...")
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_and_prepare_data() first.")
        X = self.data.drop(columns=["response"])
        y = self.data["response"]

        numeric_features = X.select_dtypes(include=["number"]).columns
        categorical_features = X.select_dtypes(
            include=["category", "object"]
        ).columns

        pre_processor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )

        self.pre_pipeline = Pipeline(steps=[("preprocessor", pre_processor)])

        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, train_size=0.8, stratify=y, random_state=78
        )

        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train_orig)
        self.y_test = self.label_encoder.transform(y_test_orig)

        X_train_processed = self.pre_pipeline.fit_transform(X_train_orig)
        self.X_test = self.pre_pipeline.transform(X_test_orig)

        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=78)
        self.X_train, self.y_train = smote.fit_resample(
            X_train_processed, y_train_encoded
        )
        print("Data preprocessing and balancing complete.")

    def train_and_evaluate(self) -> None:
        print("Training and evaluating models...")
        if (
            self.X_train is None
            or self.y_train is None
            or self.X_test is None
            or self.y_test is None
        ):
            raise ValueError(
                "Data not preprocessed. Please call preprocess_and_balance_data() first."
            )
        plt.figure(figsize=(10, 8))

        for name, model in self.models.items():
            for _ in track(range(1), description=f"Training {name}..."):
                start_time = time.time()
                model.fit(self.X_train, self.y_train)
                end_time = time.time()
                training_time = end_time - start_time

            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            new_row = pd.DataFrame(
                [
                    {
                        "Model": name,
                        "Accuracy": accuracy,
                        "ROC AUC Score": roc_auc,
                        "Training Time (s)": training_time,
                    }
                ]
            )
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

    def artifacts(self) -> None:
        print("Saving models and preprocessing artifacts...")
        for name, model in self.models.items():
            model_path = self.models_dir / f"{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_path)

        joblib.dump(self.pre_pipeline, self.models_dir / "pre_pipeline.pkl")
        joblib.dump(self.label_encoder, self.models_dir / "label_encoder.pkl")
        print(f"Artifacts saved to {self.models_dir}")

    def predict_on_new_data(
        self, new_data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        print("Making predictions on new data using the top 3 models...")
        if self.pre_pipeline is None:
            raise Exception(
                "Preprocessing pipeline is not fitted. Run preprocessing first."
            )

        if self.label_encoder is None:
            raise Exception("Label encoder is not fitted. Run preprocessing first.")

        if not isinstance(new_data, pd.DataFrame):
            raise TypeError("new_data must be a pandas DataFrame.")

        for col in new_data.select_dtypes(include=["object"]).columns:
            new_data[col] = new_data[col].astype("category")

        processed_new_data = self.pre_pipeline.transform(new_data)

        top_3_models = self.results.head(3)
        predictions: Dict[str, Dict[str, Any]] = {}

        for index, row in top_3_models.iterrows():
            model_name = row["Model"]
            model = self.models[model_name]

            prediction_encoded = model.predict(processed_new_data)
            prediction_proba = model.predict_proba(processed_new_data)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)

            predictions[model_name] = {
                "prediction": prediction[0],
                "probabilities": prediction_proba[0],
            }

            print(f"\n--- {model_name} ---")
            print(f"Prediction: {prediction[0]}")
            print(f"Prediction Probabilities: {prediction_proba[0]}")

        return predictions

    def run(self) -> pd.DataFrame:
        self.load_and_prepare_data()
        self.eda()
        self.preprocess_and_balance_data()
        self.train_and_evaluate()
        self.artifacts()
        return self.results


if __name__ == "__main__":
    # The base directory is the directory of the script.
    base_dir = Path(__file__).parent
    classifier = TD_Predictor(base_dir)
    results_df = classifier.run()

    print("\n--- Model Performance Summary ---")
    print(results_df)

    new_sample = pd.DataFrame(
        {
            "age": [30],
            "balance": [1000],
            "day": [15],
            "duration": [200],
            "campaign": [1],
            "pdays": [999],
            "previous": [0],
            "job": ["admin."],
            "marital_status": ["single"],
            "education": ["university.degree"],
            "contact": ["cellular"],
            "month": ["may"],
            "housing_loan": ["yes"],
            "personal_loan": ["no"],
            "credit_default": ["no"],
        }
    )

    classifier.predict_on_new_data(new_sample)
