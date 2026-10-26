import click
import pandas as pd
from autocure import scan, cure, train, diagnose, autopilot, serve

@click.group()
def cli():
    """Autocure: Automated Data Cleaning, Health Diagnostics & AutoML Engine."""
    pass

@cli.command()
@click.argument("file")
def scan_data(file):
    """Scan dataset for missing values, dtypes, and IQR outliers."""
    df = pd.read_csv(file)
    click.echo(scan(df))

@cli.command()
@click.argument("file")
@click.option("--target", default=None, help="Target column to check for leakage and class imbalance")
def health(file, target):
    """Deep data diagnostics: Health Score, target leakage, and ID traps."""
    df = pd.read_csv(file)
    diagnose(df, target=target)

@cli.command()
@click.argument("file")
@click.option("--out", default="clean.csv")
def fix(file, out):
    """Clean data with smart imputation and text normalization."""
    df = pd.read_csv(file)
    clean = cure(df)
    clean.to_csv(out, index=False)
    click.echo(f"Cleaned file saved: {out}")

@cli.command()
@click.argument("file")
@click.option("--target", required=True, help="Target column name")
@click.option("--mlflow/--no-mlflow", default=True, help="Track run and model versioning in MLflow (Enabled by default in command)")
@click.option("--experiment", default="autocure", help="MLflow experiment name")
@click.option("--model-name", default=None, help="Name for registered model in MLflow model registry")
@click.option("--open-dashboard", is_flag=True, default=False, help="Automatically open MLflow dashboard in browser after training")
def train_model(file, target, mlflow, experiment, model_name, open_dashboard):
    """Train models using AutoML search with automatic MLflow tracking."""
    df = pd.read_csv(file)
    result = train(
        df,
        target,
        track_mlflow=mlflow,
        experiment_name=experiment,
        registered_model_name=model_name or f"autocure_{target}",
    )
    click.echo(f"\nBest Model: {result['best_model_name']} | Score: {result['best_score']:.4f}")
    if mlflow and result.get("mlflow"):
        mf = result["mlflow"]
        click.echo(f"[MLflow] Tracking active: Experiment '{mf.get('experiment_name')}'")
        click.echo(f"[MLflow] Run ID: {mf.get('run_id')}")
        if mf.get("model_version"):
            click.echo(f"[MLflow] Registered Model '{mf.get('registered_model_name')}' -> Version {mf.get('model_version')}")
        click.echo("[MLflow] Run `autocure dashboard` to view your models and runs.\n")
        if open_dashboard:
            from autocure.tracking import launch_dashboard
            launch_dashboard()

@cli.command()
@click.argument("file")
@click.option("--target", required=True, help="Target column name")
@click.option("--mlflow/--no-mlflow", default=True, help="Track run and model versioning in MLflow (Enabled by default in command)")
@click.option("--experiment", default="autocure", help="MLflow experiment name")
@click.option("--model-name", default=None, help="Name for registered model in MLflow model registry")
@click.option("--open-dashboard", is_flag=True, default=False, help="Automatically open MLflow dashboard in browser after training")
def run_autopilot(file, target, mlflow, experiment, model_name, open_dashboard):
    """1-Click end-to-end diagnosis, auto-cure, AutoML, and feature drivers with automatic MLflow tracking."""
    df = pd.read_csv(file)
    res = autopilot(
        df,
        target=target,
        track_mlflow=mlflow,
        experiment_name=experiment,
        registered_model_name=model_name or f"autocure_{target}",
    )
    click.echo(f"\nAutopilot finished! Best: {res.best_model_name} (Score: {res.best_score:.4f})")
    if mlflow and res.mlflow:
        mf = res.mlflow
        click.echo(f"[MLflow] Tracking active: Experiment '{mf.get('experiment_name')}'")
        click.echo(f"[MLflow] Run ID: {mf.get('run_id')}")
        if mf.get("model_version"):
            click.echo(f"[MLflow] Registered Model '{mf.get('registered_model_name')}' -> Version {mf.get('model_version')}")
        click.echo("[MLflow] Run `autocure dashboard` to view your models and runs.\n")
        if open_dashboard:
            from autocure.tracking import launch_dashboard
            launch_dashboard()

@cli.command()
@click.argument("model_file")
@click.option("--port", default=8000, type=int)
def start_server(model_file, port):
    """Launch a production REST API for the given model pickle."""
    serve(model_file, port=port)

@cli.command()
@click.option("--host", default="127.0.0.1", help="Host address for MLflow UI")
@click.option("--port", default=5000, type=int, help="Port for MLflow UI")
@click.option("--browser/--no-browser", default=True, help="Automatically open default web browser")
def dashboard(host, port, browser):
    """Launch the MLflow dashboard UI to view model runs, metrics, and versions."""
    from autocure.tracking import launch_dashboard
    launch_dashboard(host=host, port=port, open_browser=browser)

@cli.command()
@click.option("--host", default="127.0.0.1", help="Host address for MLflow UI")
@click.option("--port", default=5000, type=int, help="Port for MLflow UI")
@click.option("--browser/--no-browser", default=True, help="Automatically open default web browser")
def ui(host, port, browser):
    """Alias for dashboard: Launch the MLflow UI."""
    from autocure.tracking import launch_dashboard
    launch_dashboard(host=host, port=port, open_browser=browser)

if __name__ == "__main__":
    cli()

