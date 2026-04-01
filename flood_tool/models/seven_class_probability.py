# flood_tool/models/seven_class_probability.py
"""
Seven Class Flood Probability Scale Based on Provided Labelled Samples
                                                            - Kaiqing Deng

Inputs:
  flood_tool/resources/preprocessed_data/
      - postcodes_impute_postcode_method.csv
      - district_preprocessed.csv.csv

Steps:
    1. Load Raw Data
    2. Build Preprocessing Pipeline
    3. Create RandomForest (Class_Weight Balanced)
    4. Made CV Evaluation and Confusion Matrix
    5. Save Model and Loading
    6. Output for Visualization Part
"""


from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# 1. Paths and Constants
ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "resources" / "preprocessed_data"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

POSTCODES_PREPROCESSED = RESOURCES / "postcodes_impute_postcode_method.csv"
DISTRICT_IMPUTED = RESOURCES / "district_preprocessed.csv"

MODEL_OUT = ROOT / "resources" / "seven_class_probability_classifier.pkl"
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = [1, 2, 3, 4, 5, 6, 7]
RANDOM_STATE = 42


# 2. Load Preprocessed Data
def load_data() -> pd.DataFrame:
    """Load latest preprocessed training dataset."""
    df = pd.read_csv(POSTCODES_PREPROCESSED)
    df_dist = pd.read_csv(DISTRICT_IMPUTED)

    # Merge district-level features
    df["postcodeDistrict"] = df["postcodeDistrict"].astype(str)
    df = df.merge(df_dist, on="postcodeDistrict", how="left")

    print(f"Rows loaded: {len(df)}")
    return df


# 3. Build Feature Pipeline
def build_preprocessor(df: pd.DataFrame):
    numeric_base = [
        "longitude",
        "latitude",
        "elevation",
        "distanceToWatercourse",
        "medianPrice",
        "historicallyFlooded",
    ]
    categorical_base = ["soilType", "nearestWatercourse", "localAuthority"]

    numeric_features = [c for c in numeric_base if c in df.columns]
    categorical_features = [c for c in categorical_base if c in df.columns]

    # District numeric columns
    district_numeric = []
    for col in df.columns:
        if col in numeric_features or col in categorical_features:
            continue
        if col in [
            "postcode",
            "postcodeDistrict",
            "postcodeSector",
            "riskLabel",
        ]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            district_numeric.append(col)

    numeric_features += district_numeric

    # Transformers
    numeric_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_features),
            ("cat", categorical_tf, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features + categorical_features


# 4. Train & Evaluate
def train_and_evaluate(df: pd.DataFrame, cv: bool = True):

    X = df.copy()
    y = df["riskLabel"]

    preprocessor, feat_list = build_preprocessor(df)

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])

    # CV prediction
    if cv:
        print("Running 5-fold cross-validation...")
        skf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )
        preds = cross_val_predict(pipe, X, y, cv=skf)

        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds, labels=CLASS_LABELS)
        report = classification_report(y, preds)

        # Save report
        report_path = REPORT_DIR / "seven_class_probability_cv_report.txt"
        with open(report_path, "w") as f:
            f.write("CONFUSION MATRIX:\n")
            f.write(str(cm) + "\n\n")
            f.write("CLASSIFICATION REPORT:\n")
            f.write(report + "\n\n")
            f.write(f"CV Accuracy: {acc:.5f}\n")

        print(f"CV report written to: {report_path}")

    else:
        print("Skipping cross-validation for fast training mode.")

    # Fit final model
    print("Training final model on full dataset...")
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_OUT)
    print(f"Model saved → {MODEL_OUT}")

    return pipe


# 5. Public API
class SevenClassProbability:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: Path = MODEL_OUT):
        return cls(joblib.load(path))

    def predict(self, df):
        return self.pipeline.predict(df)

    def predict_proba(self, df):
        return self.pipeline.predict_proba(df)

    def predict_dataframe(self, df, return_proba=True):
        df_out = df.copy()
        preds = self.predict(df)
        df_out["predictedRisk"] = preds

        if return_proba:
            proba = self.predict_proba(df)
            for i, cls in enumerate(CLASS_LABELS):
                df_out[f"prob_risk_{cls}"] = proba[:, i]

        return df_out


# 6. Run part
if __name__ == "__main__":
    df = load_data()
    train_and_evaluate(df)
