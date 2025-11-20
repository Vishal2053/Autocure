import click
import pandas as pd
from autocure import scan, cure, train

@click.group()
def cli():
    pass

@cli.command()
@click.argument("file")
def scan_data(file):
    df = pd.read_csv(file)
    click.echo(scan(df))

@cli.command()
@click.argument("file")
@click.option("--out", default="clean.csv")
def fix(file, out):
    df = pd.read_csv(file)
    clean = cure(df)
    clean.to_csv(out, index=False)
    click.echo(f"Cleaned file saved: {out}")

@cli.command()
@click.argument("file")
@click.option("--target", required=True)
def train_model(file, target):
    df = pd.read_csv(file)
    result = train(df, target)
    click.echo(result)

if __name__ == "__main__":
    cli()
