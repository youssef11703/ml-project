"""
pipeline_prefect.py — orchestration du pipeline ML avec Prefect (Atelier 3).

Tasks  : une par étape, chacune appelle une fonction de main.py.
Flows  : install, code, train, evaluate, all.

Lancement :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train          (alias : entrainement)
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow code
    python pipeline_prefect.py --flow install
"""

import argparse
from typing import Optional

from prefect import flow, task

import main as ml

try:  # Prefect 3 : on évite de hacher/persister les gros DataFrames entre tasks
    from prefect.cache_policies import NO_CACHE

    TASK_OPTIONS = {"cache_policy": NO_CACHE}
except ImportError:  # pragma: no cover - versions sans cache_policies
    TASK_OPTIONS = {}


# =========================================================================== #
# TASKS
# =========================================================================== #
@task(name="fetch_project", **TASK_OPTIONS)
def fetch_project_task(repo_url: str) -> str:
    """Étape 06 : récupère (clone/pull) le projet depuis le dépôt git distant."""
    return ml.fetch_project(repo_url)


@task(name="install_dependencies", **TASK_OPTIONS)
def install_dependencies_task() -> None:
    """Task 1 : installe les dépendances (requirements.txt)."""
    ml.install_dependencies()


@task(name="format_code", **TASK_OPTIONS)
def format_code_task() -> None:
    """Formatage du code (black) sur les 3 fichiers du projet."""
    ml.format_code()


@task(name="check_quality", **TASK_OPTIONS)
def check_quality_task() -> None:
    """Qualité du code (flake8) sur les 3 fichiers du projet."""
    ml.check_quality()


@task(name="check_security", **TASK_OPTIONS)
def check_security_task() -> None:
    """Sécurité du code (bandit) sur les 3 fichiers du projet."""
    ml.check_security()


@task(name="run_unit_tests", **TASK_OPTIONS)
def run_unit_tests_task() -> None:
    """Exécution des tests unitaires (pytest)."""
    ml.run_unit_tests()


@task(name="prepare_data", **TASK_OPTIONS)
def prepare_data_task(data_path: str):
    """Task 2 : préparation des données -> (X_train, X_val, X_test, y_train, y_val, y_test)."""
    return ml.prepare(data_path)


@task(name="train_model", **TASK_OPTIONS)
def train_model_task(X_train, y_train, model_name: str):
    """Task 3 : entraînement du modèle."""
    return ml.train(X_train, y_train, model_name)


@task(name="save_model", **TASK_OPTIONS)
def save_model_task(model, model_path: str) -> str:
    """Task 4 : sauvegarde du modèle."""
    return ml.save(model, model_path)


@task(name="load_model", **TASK_OPTIONS)
def load_model_task(model_path: str):
    """Task 5 : chargement du modèle sauvegardé."""
    return ml.load(model_path)


@task(name="evaluate_model", **TASK_OPTIONS)
def evaluate_model_task(model, splits, model_name: str):
    """Task 6 : évaluation du modèle sur validation et test."""
    _, X_val, X_test, _, y_val, y_test = splits
    return ml.evaluate(model, X_val, y_val, X_test, y_test, model_name)


@task(name="predict", **TASK_OPTIONS)
def predict_task(model, splits, n_samples: int = 5):
    """Prédiction sur quelques exemples du jeu de test."""
    X_test = splits[2]
    return ml.predict(model, X_test, n_samples)


# =========================================================================== #
# FLOWS
# =========================================================================== #
@flow(name="install", log_prints=True)
def install_flow() -> None:
    """Installe uniquement les dépendances."""
    install_dependencies_task()


@flow(name="code", log_prints=True)
def code_flow(skip_install: bool = False) -> None:
    """Suivi du code : (installation) -> format -> qualité -> sécurité -> tests."""
    if not skip_install:
        install_dependencies_task()
    format_code_task()
    check_quality_task()
    check_security_task()
    run_unit_tests_task()


@flow(name="train", log_prints=True)
def train_flow(
    data_path: str = ml.DEFAULT_DATA,
    model_name: str = "random_forest",
    model_path: Optional[str] = None,
):
    """Préparation des données -> entraînement -> sauvegarde du modèle.

    La sauvegarde est incluse pour que le flow `evaluate` puisse recharger le modèle.
    """
    model_path = model_path or ml.default_model_path(model_name)
    splits = prepare_data_task(data_path)
    model = train_model_task(splits[0], splits[3], model_name)
    save_model_task(model, model_path)
    return model


@flow(name="evaluate", log_prints=True)
def evaluate_flow(
    data_path: str = ml.DEFAULT_DATA,
    model_name: str = "random_forest",
    model_path: Optional[str] = None,
):
    """Chargement du modèle sauvegardé -> évaluation.

    Les données sont re-préparées avec le même découpage (graine fixe).
    """
    model_path = model_path or ml.default_model_path(model_name)
    splits = prepare_data_task(data_path)
    model = load_model_task(model_path)
    return evaluate_model_task(model, splits, model_name)


@flow(name="all", log_prints=True)
def all_flow(
    data_path: str = ml.DEFAULT_DATA,
    model_name: str = "random_forest",
    model_path: Optional[str] = None,
    skip_install: bool = False,
    repo_url: Optional[str] = None,
    n_predictions: int = 5,
):
    """Pipeline complet.

    (git) -> installation -> code (format, qualité, sécurité, tests)
          -> préparation -> entraînement -> sauvegarde -> évaluation -> prédiction.
    """
    model_path = model_path or ml.default_model_path(model_name)

    if repo_url:
        fetch_project_task(repo_url)

    code_flow(skip_install=skip_install)

    splits = prepare_data_task(data_path)
    model = train_model_task(splits[0], splits[3], model_name)
    save_model_task(model, model_path)
    results = evaluate_model_task(model, splits, model_name)
    predict_task(model, splits, n_predictions)
    return results


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestration Prefect du pipeline ML")
    parser.add_argument(
        "--flow",
        default="all",
        choices=["all", "train", "entrainement", "evaluate", "code", "install"],
        help="flow à lancer ('entrainement' est un alias de 'train')",
    )
    parser.add_argument("--data", default=ml.DEFAULT_DATA, help="chemin du CSV")
    parser.add_argument("--model", default="random_forest", choices=ml.AVAILABLE_MODELS)
    parser.add_argument("--model-path", default=None, help="fichier .joblib du modèle")
    parser.add_argument(
        "--skip-install", action="store_true", help="ne pas réinstaller les dépendances"
    )
    parser.add_argument("--repo-url", default=None, help="dépôt git à cloner (Étape 06)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow_name = "train" if args.flow == "entrainement" else args.flow

    if flow_name == "install":
        install_flow()
    elif flow_name == "code":
        code_flow(skip_install=args.skip_install)
    elif flow_name == "train":
        train_flow(args.data, args.model, args.model_path)
    elif flow_name == "evaluate":
        evaluate_flow(args.data, args.model, args.model_path)
    else:
        all_flow(
            data_path=args.data,
            model_name=args.model,
            model_path=args.model_path,
            skip_install=args.skip_install,
            repo_url=args.repo_url,
        )


if __name__ == "__main__":
    main()
