"""
deploiement_prefect.py — déploiements Prefect planifiés (Atelier 3, étape 05).

Terminal 1 : prefect server start                      # UI : http://localhost:4200
Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
             python deploiement_prefect.py
Terminal 3 : prefect deployment ls
             prefect deployment run 'all/ml-pipeline-all'
             prefect deployment run 'train/ml-pipeline-train'
"""

from prefect import serve

from pipeline_prefect import all_flow, code_flow, evaluate_flow, install_flow, train_flow

if __name__ == "__main__":
    deployments = [
        # Pipeline complet : tous les jours à 02:00
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
            parameters={"skip_install": True},
        ),
        # Contrôle du code : tous les jours à 01:00
        code_flow.to_deployment(
            name="ml-pipeline-code",
            cron="0 1 * * *",
            tags=["code-quality", "mlops"],
            parameters={"skip_install": True},
        ),
        # Déploiements à lancer manuellement
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
    ]
    serve(*deployments)
