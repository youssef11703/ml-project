"""
main.py — fonctions d'exécution du pipeline ML + point d'entrée CLI.

Chaque étape du pipeline est exposée sous forme d'une fonction simple, appelée
par les tasks Prefect de `pipeline_prefect.py`.

Exemples en ligne de commande :
    python main.py prepare
    python main.py train    --model random_forest
    python main.py evaluate --model random_forest
    python main.py predict  --model random_forest
    python main.py all      --model decision_tree
"""

import argparse
import subprocess  # nosec B404 - appels contrôlés d'outils de développement
import sys
from pathlib import Path

import pandas as pd

from model_pipeline import (
    AVAILABLE_MODELS,
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = str(BASE_DIR / "diabetic_data.csv")
# Les étapes qualité / sécurité ne ciblent QUE les fichiers du projet
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
LINE_LENGTH = "100"


def default_model_path(model_name: str) -> str:
    """Chemin par défaut du modèle sauvegardé (models/<nom>.joblib)."""
    return str(BASE_DIR / "models" / f"{model_name}.joblib")


# --------------------------------------------------------------------------- #
# Outils / environnement
# --------------------------------------------------------------------------- #
def _run(command: list, description: str) -> str:
    """Exécute une commande, affiche sa sortie et lève une erreur si elle échoue."""
    print(f"[{description}] {' '.join(command)}")
    result = subprocess.run(  # nosec B603 - liste d'arguments fixe, sans shell
        command, cwd=BASE_DIR, capture_output=True, text=True, check=False
    )
    output = (result.stdout or "") + (result.stderr or "")
    if output.strip():
        print(output.strip())
    if result.returncode != 0:
        raise RuntimeError(f"{description} : échec (code {result.returncode})")
    return result.stdout


def install_dependencies(requirements: str = "requirements.txt") -> None:
    """Installe les dépendances listées dans requirements.txt."""
    _run([sys.executable, "-m", "pip", "install", "-r", requirements], "Installation")


def fetch_project(repo_url: str, destination: str = "repo") -> str:
    """Clone (ou met à jour) le dépôt git distant du projet (Étape 06)."""
    target = BASE_DIR / destination
    if (target / ".git").exists():
        _run(["git", "-C", str(target), "pull"], "Mise à jour du dépôt")
    else:
        _run(["git", "clone", repo_url, str(target)], "Clonage du dépôt")
    return str(target)


# --------------------------------------------------------------------------- #
# Suivi du code (Étape 04) — limité aux fichiers du projet
# --------------------------------------------------------------------------- #
def format_code(files: list = None) -> None:
    """Formate le code avec black (modifie les fichiers en place)."""
    _run(
        [sys.executable, "-m", "black", "--line-length", LINE_LENGTH, *(files or PROJECT_FILES)],
        "Formatage (black)",
    )


def check_quality(files: list = None) -> None:
    """Vérifie la qualité du code avec flake8 (échoue si des erreurs)."""
    _run(
        [
            sys.executable,
            "-m",
            "flake8",
            f"--max-line-length={LINE_LENGTH}",
            *(files or PROJECT_FILES),
        ],
        "Qualité (flake8)",
    )


def check_security(files: list = None) -> None:
    """Analyse de sécurité avec bandit (échoue si problème moyen/élevé)."""
    _run(
        [sys.executable, "-m", "bandit", "-q", "-ll", *(files or PROJECT_FILES)],
        "Sécurité (bandit)",
    )


def run_unit_tests(path: str = "tests") -> None:
    """Exécute les tests unitaires avec pytest."""
    _run([sys.executable, "-m", "pytest", path, "-q"], "Tests unitaires (pytest)")


# --------------------------------------------------------------------------- #
# Étapes ML
# --------------------------------------------------------------------------- #
def prepare(data_path: str = DEFAULT_DATA):
    """Charge et prépare les données.

    Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    return prepare_data(data_path)


def train(X_train, y_train, model_name: str = "random_forest"):
    """Entraîne le modèle choisi et le retourne."""
    return train_model(X_train, y_train, model_name)


def save(model, path: str) -> str:
    """Sauvegarde le modèle et retourne le chemin du fichier."""
    return save_model(model, path)


def load(path: str):
    """Charge un modèle sauvegardé."""
    return load_model(path)


def evaluate(model, X_val, y_val, X_test, y_test, name: str = "modèle") -> pd.DataFrame:
    """Évalue le modèle sur validation et test, retourne un tableau de métriques."""
    results = pd.DataFrame(
        [
            evaluate_model(model, X_val, y_val, f"{name} (validation)", verbose=False),
            evaluate_model(model, X_test, y_test, f"{name} (test)"),
        ]
    ).set_index("Modèle")
    results = results.round(4)
    print("\nRésumé des métriques :")
    print(results.to_string())
    return results


def predict(model, X, n_samples: int = 5) -> pd.DataFrame:
    """Prédit la réadmission <30 j pour les `n_samples` premières lignes de X."""
    sample = X.head(n_samples)
    output = pd.DataFrame({"prediction": model.predict(sample)}, index=sample.index)
    if hasattr(model, "predict_proba"):
        output["proba_readmission_30j"] = model.predict_proba(sample)[:, 1].round(4)
    print("\nPrédictions :")
    print(output.to_string())
    return output


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline ML — réadmission diabète")
    parser.add_argument(
        "action",
        choices=["prepare", "train", "evaluate", "predict", "all"],
        help="prepare: vérifier les données | train: entraîner+sauvegarder | "
        "evaluate: évaluer le modèle sauvegardé | predict: prédire quelques lignes | "
        "all: tout enchaîner",
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="chemin du CSV")
    parser.add_argument("--model", default="random_forest", choices=AVAILABLE_MODELS)
    parser.add_argument(
        "--model-path", default=None, help="fichier .joblib (défaut : models/<model>.joblib)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path or default_model_path(args.model)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare(args.data)

    if args.action == "prepare":
        print("Données préparées :")
        print(f"  X_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape}")
        print(f"  Taux de réadmission <30 j (train) : {y_train.mean():.2%}")
        print(f"  Valeurs manquantes (train) : {int(X_train.isna().sum().sum())}")
        return

    if args.action in ("train", "all"):
        print(f"Entraînement du modèle : {args.model} ...")
        model = train(X_train, y_train, args.model)
        print(f"Composantes PCA retenues : {model.named_steps['pca'].n_components_}")
        print(f"Modèle sauvegardé : {save(model, model_path)}")
    else:
        model = load(model_path)
        print(f"Modèle chargé : {model_path}")

    if args.action in ("evaluate", "all"):
        evaluate(model, X_val, y_val, X_test, y_test, args.model)
    if args.action == "predict":
        predict(model, X_test)


if __name__ == "__main__":
    main()
