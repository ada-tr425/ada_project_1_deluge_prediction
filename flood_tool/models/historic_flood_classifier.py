from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# Paths for saving model and CV report
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "historic_flood_classifier.pkl"
)
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

DECISION_THRESHOLD = 0.4


def load_labelled_data() -> pd.DataFrame:
    """
    Load the imputed labelled postcode data and print basic info.
    """
    base_dir = Path(__file__).resolve().parents[1]
    data_path = (
        base_dir
        / "resources"
        / "preprocessed_data"
        / "postcodes_impute_postcode_method.csv"
    )

    df = pd.read_csv(data_path)

    # Quick sanity checks

    # print(df.head())

    return df


def build_preprocessor(df: pd.DataFrame):
    """
    Build preprocessing pipeline for numeric and categorical features.
    """
    target_col = "historicallyFlooded"

    # ID-like columns that should not be used as features
    id_cols = ["postcode", "postcodeDistrict", "postcodeSector"]

    # All remaining columns are candidate features
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

    # Split into numeric and categorical features
    numeric_features = (
        df[feature_cols].select_dtypes(include="number").columns.tolist()
    )
    categorical_features = [
        c for c in feature_cols if c not in numeric_features
    ]

    # Numeric pipeline: impute missing values with median + standardize
    numeric_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute with most frequent value + one-hot encode
    categorical_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # Combine both into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_features),
            ("cat", categorical_tf, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, feature_cols


_df_for_features = load_labelled_data()
_, FEATURE_COLS = build_preprocessor(_df_for_features)
del _df_for_features


def build_historic_rf(df: pd.DataFrame) -> Pipeline:
    if df is None:
        df = load_labelled_data()

    preprocessor, feature_cols = build_preprocessor(df)
    X = df[feature_cols].copy()
    y = df["historicallyFlooded"].astype(int)

    clf = RandomForestClassifier(
        n_estimators=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight={0: 1.0, 1: 3.0},
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    pipe.fit(X, y)

    return pipe


def train_and_evaluate(df: pd.DataFrame):
    """
    Train a RandomForest classifier with stratified CV,
    print metrics, save a text report, and save the trained model.
    """
    # Get preprocessor and feature list from Step 2
    preprocessor, feature_cols = build_preprocessor(df)

    # Features and target
    X = df[feature_cols].copy()
    y = df["historicallyFlooded"].astype(int)

    # RandomForest model
    clf = RandomForestClassifier(
        n_estimators=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight={0: 1.0, 1: 3.0},
    )

    # Full pipeline = preprocessing + model
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # cross_val_predict
    y_proba = cross_val_predict(
        pipe,
        X,
        y,
        cv=skf,
        n_jobs=-1,
        method="predict_proba",
    )[:, 1]

    # Apply global decision threshold to get class labels
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    # Metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, pos_label=1)

    print(
        f"\n Cross-validated performance on labelled data "
        f"(threshold = {DECISION_THRESHOLD}) "
    )
    # print(f"Accuracy: {acc:.3f}")
    # print(f"F1-score (class 1 = historically flooded): {f1:.3f}")

    # print("\nClassification report:")
    report_str = classification_report(y, y_pred)
    # print(report_str)

    cm = confusion_matrix(y, y_pred)
    # print("Confusion matrix:")
    # print(cm)

    # Save CV report as a text file
    report_path = REPORT_DIR / "historic_flood_classifier_cv_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Decision threshold: {DECISION_THRESHOLD:.3f}\n")
        f.write("Accuracy: {:.4f}\n".format(acc))
        f.write("F1-score (class 1): {:.4f}\n\n".format(f1))
        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification report:\n")
        f.write(report_str)

    print(f"\nCV report written to: {report_path}")

    return pipe, cm, acc, f1


def predict_postcode(postcode: str) -> pd.DataFrame:
    """
    Convenience function for demo and visualisation.

    Steps:
    1. Load the full imputed-postcode dataset.
    2. Select the row for the given postcode.
    3. Load the trained model from MODEL_PATH.
    4. Predict historic flooding and probability for this row.

    Returns
    DataFrame with the original columns plus:
    - predictedHistoricFlooded
    - predictedFloodProb  (probability of class 1)
    """
    # 1) Load full preprocessed table
    df = load_labelled_data()

    # 2) Filter the given postcode
    row = df[df["postcode"] == postcode].copy()

    if row.empty:
        print(f"No row found for postcode '{postcode}'.")
        return row

    # 3–4) Load trained pipeline and predict
    pipe = joblib.load(MODEL_PATH)
    y_proba = pipe.predict_proba(row)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    row["predictedHistoricFlooded"] = y_pred
    row["predictedFloodProb"] = y_proba

    print(
        row[
            [
                "postcode",
                "historicallyFlooded",
                "predictedHistoricFlooded",
                "predictedFloodProb",
            ]
        ]
    )

    return row


if __name__ == "__main__":

    df = load_labelled_data()
    train_and_evaluate(df)
