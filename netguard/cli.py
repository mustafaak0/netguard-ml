"""netguard command line interface (typer)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from netguard import __version__
from netguard.config import NetGuardConfig

app = typer.Typer(
    name="netguard",
    help="Network anomaly detection with an Isolation Forest + Autoencoder ensemble.",
    no_args_is_help=True,
    add_completion=False,
)


def _resolve_config(config: Path | None) -> NetGuardConfig:
    return NetGuardConfig.from_yaml(config) if config else NetGuardConfig()


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(f"netguard-ml {__version__}")


@app.command("generate-data")
def generate_data(
    samples: int = typer.Option(20_000, help="Number of flows to generate"),
    attack_ratio: float = typer.Option(0.05, help="Fraction of attack flows"),
    seed: int = typer.Option(42, help="Random seed"),
    out: Path = typer.Option(Path("data/sample.csv"), "--out", "-o", help="Output CSV path"),
) -> None:
    """Generate a synthetic KDD-style traffic dataset for experimentation."""
    from netguard.data import generate_synthetic

    df = generate_synthetic(n_samples=samples, attack_ratio=attack_ratio, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    attacks = int((df["label"] != "normal").sum())
    typer.echo(f"wrote {len(df)} flows ({attacks} attacks) -> {out}")


@app.command()
def train(
    data: Path = typer.Option(..., "--data", "-d", help="Training CSV path"),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="YAML config path (defaults to built-in config)"
    ),
    out: Path = typer.Option(Path("artifacts"), "--out", "-o", help="Artifact output directory"),
) -> None:
    """Train the ensemble and save artifacts."""
    from netguard.train import train_from_csv

    cfg = _resolve_config(config)
    metadata = train_from_csv(data, cfg, out)
    typer.echo(f"trained in {metadata['training_seconds']}s -> {out}")
    typer.echo(json.dumps(metadata.get("metrics", {}) or {"note": "unlabeled data"}, indent=2))


@app.command()
def predict(
    data: Path = typer.Option(..., "--data", "-d", help="CSV of flows to score"),
    model: Path = typer.Option(Path("artifacts"), "--model", "-m", help="Model directory"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write predictions as JSON to this path"
    ),
) -> None:
    """Score flows and print risk reports."""
    from netguard.data import load_dataset
    from netguard.predict import Predictor

    df = load_dataset(data)
    results = Predictor.load(model).predict(df)
    payload = [r.to_dict() for r in results]
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"wrote {len(payload)} predictions -> {output}")
    for i, item in enumerate(payload[:10]):
        typer.echo(
            f"[{i}] risk={item['risk_score']:3d} level={item['level']:<10} "
            f"top={item['top_features'][0]['feature'] if item['top_features'] else '-'}"
        )
    if len(payload) > 10:
        typer.echo(f"... and {len(payload) - 10} more")


@app.command()
def serve(
    model: Path = typer.Option(Path("artifacts"), "--model", "-m", help="Model directory"),
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload (development)"),
) -> None:
    """Serve the prediction API with uvicorn."""
    import os

    os.environ["NETGUARD_MODEL_DIR"] = str(model)
    import uvicorn

    uvicorn.run("netguard.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
