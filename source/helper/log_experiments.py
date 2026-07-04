from pathlib import Path
import mlflow
from sklearn import metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"


def log_experiments_mlflow(model,
                           test_dataset,
                           test_target,
                           experiment_name: str,
                           run_name: str,
                           tracking_uri: str | None = None):

    if tracking_uri is None:
        tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        y_pred = model.predict(test_dataset)

        mlflow.log_param("model", model.__class__.__name__)

        for key, value in model.get_params().items():
            mlflow.log_param(key, value)

        mlflow.log_metric("accuracy", metrics.accuracy_score(test_target, y_pred))
        mlflow.log_metric("precision", metrics.precision_score(test_target, y_pred))
        mlflow.log_metric("recall", metrics.recall_score(test_target, y_pred))
        mlflow.log_metric("f1_score", metrics.f1_score(test_target, y_pred))

        print(f"Logged run '{run_name}' in experiment '{experiment_name}'")