from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import __version__
from .paths import CONFIG_DIR, RESULTS_DIR, TEST_SETS_DIR

app = typer.Typer(
    name="mate",
    help="MATE — Model AI Throughput Evaluator. Benchmark local AI workloads.",
    no_args_is_help=True,
)
console = Console()


def _load_user_config() -> dict:
    cfg_file = CONFIG_DIR / "config.yaml"
    if cfg_file.exists():
        return yaml.safe_load(cfg_file.read_text()) or {}
    return {}


@app.command()
def run(
    workloads: list[str] = typer.Argument(..., help="Workloads: llm, image, speech"),
    engine: Optional[str] = typer.Option(None, "--engine", help="Override default engine"),
    profile: Optional[str] = typer.Option(None, "--profile", help="quick | standard | full (default: from config or 'standard')"),
    mode: str = typer.Option("closed", "--mode", help="closed | open"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for open mode (e.g. 'deepseek-r1:14b')"),
    local: bool = typer.Option(False, "--local", help="Only already-installed models"),
    runs: int = typer.Option(5, "--runs", help="Number of measurement runs"),
    warmup_runs: int = typer.Option(1, "--warmup", help="Number of warmup runs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without executing"),
    output: Optional[Path] = typer.Option(None, "--output", help="Save result YAML to file"),
) -> None:
    """Run benchmark for one or more workloads."""
    from .discovery import get_plugin_versions, load_engines, load_runtimes, load_workloads
    from .plugin import Mode
    from ._preflight import PreflightError, check_engine_available, check_model_pulled, check_vram
    from ._system import collect_system_info
    from ._submitter import get_anonymous_id
    from .schema import (
        BenchmarkResult, BenchVersions, EngineConfig,
        Measurement as SchemaMeasurement, ModelInfo, Submitter, SystemInfo, TestSetRef,
    )

    # Resolve profile from config if not explicitly given
    if profile is None:
        profile = _load_user_config().get("default_profile", "standard")

    available_workloads = load_workloads()
    for name in workloads:
        if name not in available_workloads:
            console.print(f"[red]Unknown workload: {name}[/red]")
            console.print(f"Available: {', '.join(available_workloads) or 'none installed'}")
            raise typer.Exit(1)

    try:
        bench_mode = Mode(mode)
    except ValueError:
        console.print(f"[red]Invalid mode: {mode}. Use 'closed' or 'open'.[/red]")
        raise typer.Exit(1)

    available_engines = load_engines()
    if engine:
        if engine not in available_engines:
            console.print(f"[red]Unknown engine: {engine}[/red]")
            console.print(f"Available: {', '.join(available_engines) or 'none installed'}")
            raise typer.Exit(1)
        selected_engine = available_engines[engine]
    else:
        if not available_engines:
            console.print("[red]No engine plugins installed.[/red]")
            raise typer.Exit(1)
        selected_engine = next(iter(available_engines.values()))

    if dry_run:
        console.print(
            f"[bold]Plan[/bold]  workloads={', '.join(workloads)}  "
            f"engine={selected_engine.name}  profile={profile}  "
            f"mode={mode}  runs={runs}  warmup={warmup_runs}"
        )
        for wl_name in workloads:
            wl = available_workloads[wl_name]
            if profile in wl.profiles:
                cfg = wl.profiles[profile]
                models = (
                    wl.required_models(profile)
                    if hasattr(wl, "required_models")
                    else [cfg.reference_engine_config.get("model", "?")]
                )
                console.print(
                    f"  {wl_name}/{profile}: models={', '.join(models)}  "
                    f"~{cfg.estimated_runtime_seconds//60} min  "
                    f"{cfg.vram_required_gb:.1f} GB VRAM"
                )
        return

    available_runtimes = load_runtimes()

    try:
        check_engine_available(selected_engine)
    except PreflightError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]OK[/green] Engine [bold]{selected_engine.name}[/bold] reachable  "
        f"[dim](v{selected_engine.version()})[/dim]"
    )

    result_files: list[Path] = []

    for wl_name in workloads:
        workload = available_workloads[wl_name]

        if profile not in workload.profiles:
            console.print(
                f"[red]Profile '{profile}' not available for workload '{wl_name}'. "
                f"Available: {', '.join(workload.profiles)}[/red]"
            )
            raise typer.Exit(1)

        prof_cfg = workload.profiles[profile]

        # Determine all models needed for this profile
        if bench_mode == Mode.OPEN:
            if not model:
                console.print(
                    f"[red]--model is required for open mode "
                    f"(e.g. --model deepseek-r1:14b)[/red]"
                )
                raise typer.Exit(1)
            required = [model]
        else:
            required = (
                workload.required_models(profile)
                if hasattr(workload, "required_models")
                else [prof_cfg.reference_engine_config.get("model", "")]
            )

        # Model check / pull for each required model
        for model_name in required:
            try:
                model_available = check_model_pulled(
                    selected_engine, model_name, local_only=local
                )
            except PreflightError as e:
                console.print(f"[red]{e}[/red]")
                raise typer.Exit(1)

            if not model_available:
                if not hasattr(selected_engine, "pull"):
                    console.print(
                        f"[red]Model '{model_name}' not found and engine "
                        f"'{selected_engine.name}' doesn't support pulling.[/red]"
                    )
                    raise typer.Exit(1)
                with console.status(f"Pulling [bold]{model_name}[/bold]..."):
                    selected_engine.pull(model_name)
                console.print(f"[green]OK[/green] Pulled {model_name}")
            else:
                console.print(f"[green]OK[/green] Model [bold]{model_name}[/bold] available")

        # VRAM check
        sys_info = collect_system_info(available_runtimes)
        check_vram(sys_info.vram_gb, prof_cfg.vram_required_gb, profile)

        # Setup test set(s)
        if bench_mode == Mode.OPEN:
            workload.setup_open(profile, {"model": model})
        else:
            workload.setup_closed(profile)

        # Run
        console.print(
            f"\nRunning [bold]{wl_name}[/bold] / [bold]{profile}[/bold]  "
            f"({runs} runs + {warmup_runs} warmup)..."
        )
        with console.status("Benchmarking..."):
            plugin_m = workload.run(
                profile=profile,
                mode=bench_mode,
                engine=selected_engine,
                runs=runs,
                warmup_runs=warmup_runs,
            )

        measurement = SchemaMeasurement(**asdict(plugin_m))

        if measurement.throttling_detected:
            console.print(
                "[yellow]WARN: Throttling detected -- token rate varied >15% across runs. "
                "Result flagged.[/yellow]"
            )

        # Model info — use primary (first) model for the result header
        primary_model = required[0]
        if hasattr(selected_engine, "model_info"):
            model_info = selected_engine.model_info(primary_model)
        else:
            model_info = ModelInfo(
                name=primary_model, source="local", source_ref=primary_model
            )

        # Test set hash
        ts_hash = "unknown"
        if hasattr(workload, "test_set_hash"):
            ts_hash = workload.test_set_hash(profile)

        # Build result
        result = BenchmarkResult.new(
            mode=bench_mode.value,
            workload=wl_name,
            engine=selected_engine.name,
            engine_version=selected_engine.version(),
            model=model_info,
            profile=profile,
            test_set=TestSetRef(
                hash=ts_hash,
                id=getattr(prof_cfg, "test_set_id", None),
            ),
            engine_config=EngineConfig(
                is_reference=(bench_mode == Mode.CLOSED),
                settings=(
                    prof_cfg.reference_engine_config
                    if bench_mode == Mode.CLOSED
                    else {"engine": selected_engine.name, "model": model}
                ),
            ),
            system=sys_info,
            bench=BenchVersions(
                core_version=__version__,
                plugin_versions=get_plugin_versions(),
            ),
            measurement=measurement,
            submitter=Submitter(anonymous_id=get_anonymous_id()),
        )
        result.compute_integrity()

        # Save
        yaml_str = result.to_yaml()
        if output and len(workloads) == 1:
            out_path = output
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = RESULTS_DIR / f"{ts}-{wl_name}-{profile}.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_str)
        result_files.append(out_path)

        # Summary table
        table = Table(title=f"Result: {wl_name}/{profile}", show_header=False)
        table.add_column("Key", style="dim")
        table.add_column("Value", style="bold")
        _add_tps_rows(table, measurement)
        table.add_row("Runs", f"{runs} + {warmup_runs} warmup")
        table.add_row("Throttling", "yes" if measurement.throttling_detected else "no")
        table.add_row("Saved to", str(out_path))
        console.print(table)

    console.print(f"\n[green]Done.[/green] Run [bold]mate submit {result_files[0]}[/bold] to share.")


def _add_tps_rows(table: Table, measurement: SchemaMeasurement) -> None:
    """Add tokens/s rows to a summary table, handling both flat and per-model median dicts."""
    from .schema import Measurement as SchemaMeasurement  # local import avoids circular

    flat_tps = measurement.median.get("tokens_per_second")
    if flat_tps is not None:
        tps_sd = measurement.std_dev.get("tokens_per_second", 0.0)
        table.add_row("Tokens/s (median)", f"{flat_tps:.1f}")
        table.add_row("Tokens/s (std dev)", f"{tps_sd:.1f}")
    else:
        for model_key, stats in measurement.median.items():
            tps = stats.get("tokens_per_second", 0.0) if isinstance(stats, dict) else 0.0
            table.add_row(f"Tokens/s {model_key}", f"{tps:.1f}")


@app.command()
def submit(
    result_file: Optional[Path] = typer.Argument(None, help="Result YAML to submit (default: latest)"),
    discord: bool = typer.Option(False, "--discord", help="Show YAML for Discord submission"),
    print_yaml: bool = typer.Option(False, "--print", help="Print YAML to stdout"),
) -> None:
    """Submit a benchmark result to the leaderboard."""
    import jsonschema

    # Resolve file
    if result_file is None:
        if not RESULTS_DIR.exists():
            console.print("[red]No result files found. Run a benchmark first.[/red]")
            raise typer.Exit(1)
        yamls = sorted(RESULTS_DIR.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not yamls:
            console.print("[red]No result files found in results directory. Run a benchmark first.[/red]")
            raise typer.Exit(1)
        result_file = yamls[0]
        console.print(f"Using latest result: [bold]{result_file.name}[/bold]")

    if not result_file.exists():
        console.print(f"[red]File not found: {result_file}[/red]")
        raise typer.Exit(1)

    yaml_content = result_file.read_text()

    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML: {e}[/red]")
        raise typer.Exit(1)

    from .schema import _load_json_schema
    try:
        schema = _load_json_schema(data.get("schema_version", 1))
        jsonschema.validate(data, schema)
        console.print("[green]OK[/green] Result validated")
    except jsonschema.ValidationError as e:
        console.print(f"[red]Validation error: {e.message}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Schema load error: {e}[/red]")
        raise typer.Exit(1)

    if print_yaml:
        console.print(yaml_content)
        return

    if discord:
        console.print("[bold]YAML content (paste into Discord #submissions):[/bold]\n")
        console.print(yaml_content)
        console.print(
            "\n[bold cyan]Discord submission:[/bold cyan] "
            "Paste the YAML above into the #submissions channel.\n"
            "[dim](Discord server link will be added once the server is live)[/dim]"
        )
        return

    import httpx

    submit_url = "https://mate-bench-api.benjamin-faeuster.workers.dev/submit/cli"
    try:
        with console.status("Submitting..."):
            response = httpx.post(
                submit_url,
                content=yaml_content.encode(),
                headers={"Content-Type": "application/x-yaml"},
                timeout=30.0,
            )
        if response.status_code == 201:
            body = response.json()
            console.print(f"[green]OK[/green] Submitted — ID: [bold]{body['id']}[/bold]")
        else:
            body = response.json()
            console.print(f"[red]Submission failed ({response.status_code}): {body.get('error')}[/red]")
            raise typer.Exit(1)
    except httpx.RequestError as e:
        console.print(f"[red]Network error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def cleanup(
    workload: Optional[str] = typer.Argument(None, help="Workload to clean (default: all)"),
) -> None:
    """Remove cached test sets."""
    from .discovery import load_workloads

    available = load_workloads()

    if workload is not None and workload not in available:
        console.print(f"[red]Unknown workload: {workload}[/red]")
        console.print(f"Available: {', '.join(available) or 'none installed'}")
        raise typer.Exit(1)

    targets = {workload: available[workload]} if workload else available

    if not targets:
        console.print("[dim]No workloads installed.[/dim]")
        return

    for name, wl in targets.items():
        for prof in wl.profiles:
            wl.cleanup(prof)
        console.print(f"[green]OK[/green] Cleaned {name}")


@app.command()
def status() -> None:
    """Show installed plugins, cached models, disk usage, recent results."""
    from .discovery import load_engines, load_runtimes, load_workloads

    console.print(f"[bold]mate-bench v{__version__}[/bold]\n")

    for label, loader in [
        ("Workloads", load_workloads),
        ("Engines", load_engines),
        ("Runtimes", load_runtimes),
    ]:
        plugins = loader()
        table = Table(title=label, show_header=True, header_style="bold")
        table.add_column("Name")
        table.add_column("API version")
        for name, plugin in plugins.items():
            table.add_row(name, str(plugin.manifest.api_version))
        if not plugins:
            table.add_row("[dim]none installed[/dim]", "")
        console.print(table)
        console.print()


@app.command()
def config() -> None:
    """Interactive configuration (default profile, default engine)."""
    existing = _load_user_config()

    console.print(f"[bold]mate-bench configuration[/bold]  [dim]{CONFIG_DIR / 'config.yaml'}[/dim]\n")

    default_profile = typer.prompt(
        "Default profile",
        default=existing.get("default_profile", "standard"),
    )

    new_cfg = {**existing, "default_profile": default_profile}
    cfg_file = CONFIG_DIR / "config.yaml"
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(yaml.dump(new_cfg))
    console.print(f"[green]OK[/green] Saved to {cfg_file}")


@app.command(name="list-engines")
def list_engines() -> None:
    """Show installed engine plugins."""
    from .discovery import load_engines

    plugins = load_engines()
    for name, plugin in plugins.items():
        runtimes = ", ".join(plugin.supported_runtimes)
        try:
            avail = plugin.is_available()
            label = "[green]available[/green]" if avail else "[red]not available[/red]"
        except NotImplementedError:
            avail = False
            label = "[dim]detection not implemented[/dim]"
        console.print(f"[bold]{name}[/bold]  {label}  runtimes: {runtimes}")
        if avail:
            try:
                console.print(f"  version: {plugin.version()}")
                models = plugin.list_models()
                if models:
                    names = ", ".join(m.get("name", "?") for m in models[:5])
                    suffix = f"  [dim]+{len(models)-5} more[/dim]" if len(models) > 5 else ""
                    console.print(f"  models:  {names}{suffix}")
                else:
                    console.print("  models:  [dim]none pulled yet[/dim]")
            except Exception as e:
                console.print(f"  [red]{e}[/red]")
    if not plugins:
        console.print("[dim]No engine plugins installed.[/dim]")


@app.command(name="list-available")
def list_available() -> None:
    """Show available plugins on PyPI (mate-engine-*, mate-runtime-*, mate-workload-*)."""
    console.print("[yellow]Not yet implemented.[/yellow]")


@app.command(name="list-workloads")
def list_workloads() -> None:
    """Show installed workload plugins."""
    from .discovery import load_workloads

    plugins = load_workloads()
    for name, plugin in plugins.items():
        profiles = ", ".join(plugin.profiles)
        console.print(f"[green]{name}[/green]  profiles: {profiles or 'none'}")
    if not plugins:
        console.print("[dim]No workload plugins installed.[/dim]")


@app.command(name="list-runtimes")
def list_runtimes() -> None:
    """Show installed runtime plugins."""
    from .discovery import load_runtimes

    plugins = load_runtimes()
    for name, plugin in plugins.items():
        try:
            avail = plugin.is_available()
            label = "[green]available[/green]" if avail else "[red]not available[/red]"
        except NotImplementedError:
            label = "[dim]detection not implemented[/dim]"
            avail = False
        console.print(f"[bold]{name}[/bold]  {label}")
        if avail:
            try:
                info = plugin.gpu_info()
                console.print(f"  GPU:    {info.get('gpu_name')} ({info.get('gpu_chip')})")
                console.print(f"  VRAM:   {info.get('vram_gb')} GB")
                console.print(f"  ROCm:   {info.get('runtime')}")
                console.print(f"  Driver: {info.get('driver')}")
                if not info.get("_gpu_name_known", True):
                    console.print("  [yellow]WARN: Unknown chip -- name not normalized, flagged for review[/yellow]")
            except Exception as e:
                console.print(f"  [red]gpu_info() failed: {e}[/red]")
    if not plugins:
        console.print("[dim]No runtime plugins installed.[/dim]")


@app.command(name="list-test-sets")
def list_test_sets() -> None:
    """Show cached test sets."""
    if not TEST_SETS_DIR.exists() or not list(TEST_SETS_DIR.iterdir()):
        console.print("[dim]No test sets cached.[/dim]")
        return
    for path in sorted(TEST_SETS_DIR.iterdir()):
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
        console.print(f"  {path.name}  [dim]{size // 1024 // 1024} MB[/dim]")


if __name__ == "__main__":
    app()
