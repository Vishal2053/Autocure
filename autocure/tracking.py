import os
import sys
import time
import threading
import webbrowser
import subprocess
from typing import Optional, Dict, Any


def is_mlflow_available() -> bool:
    """Check if MLflow is installed and importable."""
    try:
        import mlflow
        import mlflow.sklearn
        return True
    except ImportError:
        return False


def get_tracking_uri() -> str:
    """Get the current tracking URI, defaulting to local ./mlruns."""
    if not is_mlflow_available():
        return "./mlruns"
    import mlflow
    return mlflow.get_tracking_uri()


def log_autocure_run(
    train_result: Dict[str, Any],
    df=None,
    target: Optional[str] = None,
    task: str = "classification",
    experiment_name: str = "autocure",
    registered_model_name: Optional[str] = None,
    X_sample=None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Log training results, parameters, metrics, and register model version in MLflow.
    """
    if not is_mlflow_available():
        print("[MLflow] Warning: mlflow is not installed. Skipping MLflow tracking.")
        return {}

    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment(experiment_name)

    best_name = train_result.get("best_model_name", "model")
    if not run_name:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{best_name}_{timestamp}"

    if not registered_model_name and target:
        registered_model_name = f"autocure_{target}"
    elif not registered_model_name:
        registered_model_name = f"autocure_{best_name}"

    version = None
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        # 1. Tags
        mlflow.set_tag("source", "autocure")
        mlflow.set_tag("task", task)
        mlflow.set_tag("target_column", str(target) if target else "unknown")

        # 2. Dataset params
        if df is not None:
            mlflow.log_param("dataset_rows", len(df))
            mlflow.log_param("dataset_cols", df.shape[1])

        # 3. Model info & hyperparameters
        mlflow.log_param("best_model_name", best_name)
        mlflow.log_param("task", task)
        if target:
            mlflow.log_param("target", target)

        best_params = train_result.get("best_params") or {}
        for param_key, param_val in best_params.items():
            clean_key = param_key.replace("clf__", "")
            try:
                mlflow.log_param(clean_key, param_val)
            except Exception:
                pass

        # 4. Metrics
        best_score = train_result.get("best_score")
        if best_score is not None:
            metric_name = "accuracy" if task == "classification" else "r2_score"
            mlflow.log_metric(metric_name, float(best_score))
            mlflow.log_metric("best_score", float(best_score))

        leaderboard = train_result.get("leaderboard") or []
        for item in leaderboard:
            m_name = item.get("model")
            m_score = item.get("score")
            if m_name and m_score is not None:
                try:
                    mlflow.log_metric(f"score_{m_name}", float(m_score))
                except Exception:
                    pass

        # 5. Log Model Artifact & Register Model Version
        best_model = train_result.get("best_model")
        if best_model is not None:
            input_example = None
            signature = None
            if X_sample is not None and len(X_sample) > 0:
                input_example = X_sample.iloc[:5] if hasattr(X_sample, "iloc") else X_sample[:5]
                try:
                    from mlflow.models.signature import infer_signature
                    preds = best_model.predict(input_example)
                    signature = infer_signature(input_example, preds)
                except Exception:
                    pass

            try:
                model_info = mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    registered_model_name=registered_model_name,
                    input_example=input_example,
                    signature=signature,
                )
                version = getattr(model_info, "registered_model_version", None)
            except Exception as e:
                try:
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature,
                    )
                except Exception:
                    pass
                print(f"[MLflow] Model logged without registry: {e}")

        if not version and registered_model_name:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                latest = client.get_latest_versions(registered_model_name)
                if latest:
                    version = latest[-1].version
            except Exception:
                pass

    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
        "model_version": str(version) if version else None,
        "tracking_uri": get_tracking_uri(),
    }


def launch_dashboard(host: str = "127.0.0.1", port: int = 5000, open_browser: bool = True):
    """
    Launch the MLflow UI dashboard server and optionally open the browser.
    """
    if not is_mlflow_available():
        print("\n[Error] MLflow is not installed. Please install it with: pip install mlflow")
        return

    url = f"http://{host}:{port}"
    print("=" * 65)
    print("                AUTOCURE MLFLOW DASHBOARD")
    print("=" * 65)
    print(f"  Dashboard URL  : {url}")
    print(f"  Tracking Store : {get_tracking_uri()}")
    print("  Models & Runs  : View tracked experiments and model versions")
    print("  Press Ctrl+C to stop the dashboard server.")
    print("=" * 65 + "\n")

    if open_browser:
        def _open():
            time.sleep(1.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        t = threading.Thread(target=_open, daemon=True)
        t.start()

    cmd = [sys.executable, "-m", "mlflow", "ui", "--host", host, "--port", str(port)]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[MLflow] Dashboard stopped.")
