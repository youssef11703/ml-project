from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "readmitted"
ID_COLS = ["encounter_id", "patient_nbr"]
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]
VALID_TARGET_LABELS = {"NO", ">30", "<30"}
TARGET_MAPPING = {"NO": 0, ">30": 0, "<30": 1}  # 1 = réadmis < 30 jours
AGE_MAPPING = {
    "[0-10)": 0,
    "[10-20)": 1,
    "[20-30)": 2,
    "[30-40)": 3,
    "[40-50)": 4,
    "[50-60)": 5,
    "[60-70)": 6,
    "[70-80)": 7,
    "[80-90)": 8,
    "[90-100)": 9,
}
GENDER_MAPPING = {"Female": 0, "Male": 1, "Unknown": 2}
BINARY_MAPPINGS = {
    "change": {"No": 0, "Ch": 1},
    "diabetesMed": {"No": 0, "Yes": 1},
}
ADMIN_ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
AVAILABLE_MODELS = ("decision_tree", "knn", "random_forest")
CLASS_NAMES = ["Autre issue (0)", "Réadmis <30 j (1)"]


# ----------------------------- 1. Préparation ----------------------------- #
def categorize_diagnosis(code) -> str:
    """Regroupe un code diagnostic CIM-9 en grande catégorie médicale."""
    code = str(code)
    if code.startswith(("V", "E")):
        return "External_Cause"
    try:
        value = float(code)
    except ValueError:
        return "Unknown"

    bounds = [
        (1, 140, "Infectious_Parasitic"),
        (140, 240, "Neoplasms"),
        (240, 280, "Endocrine_Metabolic"),
        (280, 290, "Blood_Immune"),
        (290, 320, "Mental_Disorders"),
        (320, 390, "Nervous_Sense"),
        (390, 460, "Circulatory"),
        (460, 520, "Respiratory"),
        (520, 580, "Digestive"),
        (580, 630, "Genitourinary"),
        (630, 680, "Pregnancy"),
        (680, 710, "Skin_Subcutaneous"),
        (710, 740, "Musculoskeletal"),
        (740, 760, "Congenital"),
        (760, 780, "Perinatal"),
        (780, 800, "Symptoms_Signs"),
        (800, 1000, "Injury_Poisoning"),
    ]
    for low, high, label in bounds:
        if low <= value < high:
            return label
    return "Other"


def _text_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes texte (compatible pandas 2 'object' et pandas 3 'str')."""
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage et transformations déterministes (sans apprentissage)."""
    df = df.copy()

    for col in _text_columns(df):
        df[col] = df[col].str.strip()

    for col in ["max_glu_serum", "A1Cresult"]:
        if col in df.columns:
            df[col] = df[col].fillna("Not tested")

    df["gender"] = df["gender"].replace("Unknown/Invalid", "Unknown")
    df = df.drop(columns=["weight"], errors="ignore")

    for col in DIAG_COLS + ["race", "medical_specialty", "payer_code"]:
        if col in df.columns:
            df[col] = df[col].replace("?", "Unknown")

    df = df.drop(columns=ID_COLS, errors="ignore")

    df["gender"] = df["gender"].map(GENDER_MAPPING)
    df["age"] = df["age"].map(AGE_MAPPING)
    for col, mapping in BINARY_MAPPINGS.items():
        df[col] = df[col].map(mapping)

    for col in DIAG_COLS:
        df[col] = df[col].apply(categorize_diagnosis)

    for col in ADMIN_ID_COLS:
        df[col] = df[col].astype(str)

    return df


def prepare_data(
    csv_path: str, test_size: float = 0.20, val_size: float = 0.20, random_state: int = RANDOM_STATE
):
    """Charge, nettoie et découpe en train/val/test (stratifié).

    Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier de données introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    for col in _text_columns(df):
        df[col] = df[col].str.strip()
    df = df.drop_duplicates()
    df = df[df[TARGET].isin(VALID_TARGET_LABELS)].copy()

    y = df[TARGET].map(TARGET_MAPPING)
    if y.isna().any():
        raise ValueError("Valeurs inconnues dans la colonne cible 'readmitted'.")
    y = y.astype(int)

    X = clean_and_engineer(df.drop(columns=[TARGET]))
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant_cols)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------ 2. Entraînement ---------------------------- #
def _build_classifier(model_name: str):
    """Classifieur avec les hyperparamètres du notebook."""
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        )
    if model_name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features="log2",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Modèle inconnu : '{model_name}'. Choix : {', '.join(AVAILABLE_MODELS)}")


def build_pipeline(X_train: pd.DataFrame, model_name: str = "random_forest") -> Pipeline:
    """OneHotEncoder -> StandardScaler -> PCA(90 %) -> classifieur."""
    cat_cols = _text_columns(X_train)
    encoder = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("encoder", encoder),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.90, random_state=RANDOM_STATE)),
            ("classifier", _build_classifier(model_name)),
        ]
    )


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, model_name: str = "random_forest"
) -> Pipeline:
    """Entraîne le pipeline complet sur le jeu d'entraînement."""
    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train n'ont pas la même longueur.")
    pipeline = build_pipeline(X_train, model_name)
    pipeline.fit(X_train, y_train)
    return pipeline


# ------------------------------- 3. Évaluation ----------------------------- #
def evaluate_model(model, X, y_true, name: str = "Modèle", verbose: bool = True) -> dict:
    """Calcule les métriques (accuracy, precision/recall/F1, rappel classe 1)."""
    y_pred = model.predict(X)
    metrics = {
        "Modèle": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall classe 1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 classe 1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if verbose:
        print(f"\n===== {name} =====")
        print(
            classification_report(
                y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0
            )
        )
        print("Matrice de confusion (lignes = réel, colonnes = prédit) :")
        print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    return metrics


# ---------------------------- 4. Sauvegarde / chargement ------------------- #
def save_model(model, path: str = "models/model.joblib") -> str:
    """Sauvegarde le pipeline avec joblib (crée le dossier si besoin)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str = "models/model.joblib"):
    """Charge un modèle sauvegardé."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modèle introuvable : {path}. Lancez d'abord : python main.py train"
        )
    return joblib.load(path)
