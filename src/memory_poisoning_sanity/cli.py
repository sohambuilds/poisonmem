from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from .data import generate_data, load_banks, read_jsonl, validate_data
from .experiment import ExperimentRunner
from .metrics import analyze as analyze_results
from .metrics import preflight_gate
from .models import ExperimentConfig, Query
from .openrouter import OpenRouterClient
from .retrieval import retrieval_input_fingerprint, run_retrieval_analysis

app = typer.Typer(
    no_args_is_help=True,
    help="Run the controlled persistent-memory retrieval-poisoning sanity check.",
)

RootOption = Annotated[
    Path,
    typer.Option("--root", help="Project root containing config.yaml, data/, and results/."),
]
ConfigOption = Annotated[
    Path,
    typer.Option("--config", help="Experiment configuration, relative to --root if not absolute."),
]


def resolve(root: Path, config: Path) -> tuple[Path, ExperimentConfig]:
    root = root.expanduser().resolve()
    config_path = config if config.is_absolute() else root / config
    return root, ExperimentConfig.load(config_path)


def require_paid_confirmation(confirmed: bool) -> str:
    if not confirmed:
        raise typer.BadParameter(
            "Paid OpenRouter calls require --confirm-paid-calls. The complete run makes up to "
            "3,648 model calls, plus one malformed-output retry per call when needed."
        )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise typer.BadParameter("OPENROUTER_API_KEY is not set")
    return api_key


def ensure_data(root: Path) -> None:
    required = {
        "clean_memories.jsonl",
        "lexical_poison_memories.jsonl",
        "semantic_poison_memories.jsonl",
        "queries.jsonl",
    }
    if not required.issubset({path.name for path in (root / "data").glob("*.jsonl")}):
        generate_data(root / "data")
    validate_data(root / "data")


def ensure_retrieval(root: Path, config: ExperimentConfig) -> pd.DataFrame:
    path = root / "results" / "retrieval_results.parquet"
    banks = load_banks(root / "data")
    queries = read_jsonl(root / "data" / "queries.jsonl", Query)
    fingerprint = retrieval_input_fingerprint(banks, queries, config.retrieval)
    if path.exists():
        existing = pd.read_parquet(path)
        observed = set(existing.get("input_fingerprint", []))
        if observed == {fingerprint}:
            return existing
        raise ValueError(
            "retrieval_results.parquet does not match the current data/config; rerun retrieval "
            "with --force"
        )
    return run_retrieval_analysis(
        banks,
        queries,
        config.retrieval,
        root / ".cache" / "embeddings",
        path,
    )


@app.command("init-data")
def init_data(
    root: RootOption = Path("."),
    config: ConfigOption = Path("config.yaml"),
) -> None:
    """Generate the fixed synthetic queries and three memory banks."""
    root, _ = resolve(root, config)
    generate_data(root / "data")
    summary = validate_data(root / "data")
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("validate-data")
def validate_data_command(
    root: RootOption = Path("."),
    config: ConfigOption = Path("config.yaml"),
) -> None:
    """Validate counts, domains, IDs, action labels, and poison text constraints."""
    root, _ = resolve(root, config)
    typer.echo(json.dumps(validate_data(root / "data"), indent=2, sort_keys=True))


@app.command("retrieval")
def retrieval_command(
    root: RootOption = Path("."),
    config: ConfigOption = Path("config.yaml"),
    force: Annotated[bool, typer.Option("--force", help="Recompute cached retrieval output.")] = False,
) -> None:
    """Run Stage C locally (no LLM calls) and write retrieval_results.parquet."""
    root, experiment_config = resolve(root, config)
    ensure_data(root)
    output = root / "results" / "retrieval_results.parquet"
    if output.exists() and force:
        output.unlink()
    frame = ensure_retrieval(root, experiment_config)
    typer.echo(f"Wrote {len(frame)} rows to {output}")


async def _validate_providers(
    root: Path, config: ExperimentConfig, api_key: str
) -> list[dict[str, object]]:
    async with OpenRouterClient(config, api_key) as client:
        checks = await asyncio.gather(
            *(client.validate_provider(model) for model in config.models)
        )
    output = root / "results" / "provider_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checks


@app.command("validate-providers")
def validate_providers_command(
    root: RootOption = Path("."),
    config: ConfigOption = Path("config.yaml"),
) -> None:
    """Verify the current OpenRouter endpoint slugs without making model calls."""
    root, experiment_config = resolve(root, config)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise typer.BadParameter("OPENROUTER_API_KEY is not set")
    checks = asyncio.run(_validate_providers(root, experiment_config, api_key))
    typer.echo(json.dumps(checks, indent=2, sort_keys=True))


async def _run_stage(
    root: Path,
    config: ExperimentConfig,
    api_key: str,
    stage: str,
    retrieval: pd.DataFrame | None = None,
) -> dict[str, int]:
    async with OpenRouterClient(config, api_key) as client:
        runner = ExperimentRunner(root, config, client)
        if stage == "preflight":
            specs = runner.preflight_specs()
        elif stage == "oracle":
            specs = runner.oracle_specs()
        elif stage == "full":
            if retrieval is None:
                raise ValueError("Full stage requires retrieval results")
            specs = runner.full_specs(retrieval)
        else:
            raise ValueError(f"Unknown stage {stage}")
        return await runner.run_specs(specs)


@app.command("preflight")
def preflight_command(
    root: RootOption = Path("."),
    config: ConfigOption = Path("config.yaml"),
    confirm_paid_calls: Annotated[bool, typer.Option("--confirm-paid-calls")] = False,
) -> None:
    """Run Stage A (384 calls) and print the validity gate."""
    root, experiment_config = resolve(root, config)
    ensure_data(root)
    api_key = require_paid_confirmation(confirm_paid_calls)
    summary = asyncio.run(_run_stage(root, experiment_config, api_key, "preflight"))
    gate = preflight_gate(root, experiment_config)
    (root / "results" / "preflight_metrics.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps({"execution": summary, "validity_gate": gate}, indent=2, sort_keys=True))


@app.command("oracle")
def oracle_command(
    root: RootOption = Path("."),
    config: ConfigOption = Path("config.yaml"),
    confirm_paid_calls: Annotated[bool, typer.Option("--confirm-paid-calls")] = False,
) -> None:
    """Run Stage B oracle influence (384 calls)."""
    root, experiment_config = resolve(root, config)
    ensure_data(root)
    gate = preflight_gate(root, experiment_config)
    if not gate["passed"]:
        raise typer.BadParameter("Stage B is blocked until the current Stage A validity gate passes")
    api_key = require_paid_confirmation(confirm_paid_calls)
    summary = asyncio.run(_run_stage(root, experiment_config, api_key, "oracle"))
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("full")
def full_command(
    root: RootOption = Path("."),
    config: ConfigOption = Path("config.yaml"),
    confirm_paid_calls: Annotated[bool, typer.Option("--confirm-paid-calls")] = False,
) -> None:
    """Run Stage D retrieval-to-action matrix (2,880 calls)."""
    root, experiment_config = resolve(root, config)
    ensure_data(root)
    gate = preflight_gate(root, experiment_config)
    if not gate["passed"]:
        raise typer.BadParameter("Stage D is blocked until the current Stage A validity gate passes")
    retrieval = ensure_retrieval(root, experiment_config)
    api_key = require_paid_confirmation(confirm_paid_calls)
    summary = asyncio.run(_run_stage(root, experiment_config, api_key, "full", retrieval))
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("analyze")
def analyze_command(
    root: RootOption = Path("."),
    config: ConfigOption = Path("config.yaml"),
) -> None:
    """Compute deterministic metrics, paired bootstrap intervals, and the fixed verdict."""
    root, experiment_config = resolve(root, config)
    ensure_data(root)
    _, decision = analyze_results(root, experiment_config)
    typer.echo(json.dumps(decision, indent=2, sort_keys=True))


@app.command("run")
def run_command(
    root: RootOption = Path("."),
    config: ConfigOption = Path("config.yaml"),
    confirm_paid_calls: Annotated[bool, typer.Option("--confirm-paid-calls")] = False,
) -> None:
    """Run all stages, stopping before poisoning calls if Stage A is invalid."""
    root, experiment_config = resolve(root, config)
    api_key = require_paid_confirmation(confirm_paid_calls)
    ensure_data(root)
    provider_checks = asyncio.run(_validate_providers(root, experiment_config, api_key))
    preflight = asyncio.run(_run_stage(root, experiment_config, api_key, "preflight"))
    gate = preflight_gate(root, experiment_config)
    if not gate["passed"]:
        _, decision = analyze_results(root, experiment_config)
        typer.echo(
            json.dumps(
                {
                    "providers": provider_checks,
                    "preflight": preflight,
                    "validity_gate": gate,
                    "decision": decision,
                    "stopped_before_poisoning_stages": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise typer.Exit(code=2)
    oracle = asyncio.run(_run_stage(root, experiment_config, api_key, "oracle"))
    retrieval = ensure_retrieval(root, experiment_config)
    full = asyncio.run(_run_stage(root, experiment_config, api_key, "full", retrieval))
    _, decision = analyze_results(root, experiment_config)
    typer.echo(
        json.dumps(
            {
                "providers": provider_checks,
                "preflight": preflight,
                "oracle": oracle,
                "full": full,
                "decision": decision,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
