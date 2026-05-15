from __future__ import annotations

from importlib.metadata import entry_points

from packaging.specifiers import SpecifierSet

_PLUGIN_GROUPS = [
    "mate_bench.workload",
    "mate_bench.engine",
    "mate_bench.runtime",
]


def get_plugin_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for group in _PLUGIN_GROUPS:
        for ep in entry_points(group=group):
            if ep.dist is not None:
                versions[ep.name] = ep.dist.version
    return versions


from . import __version__  # noqa: E402
from .plugin import EnginePlugin, PluginManifest, RuntimePlugin, WorkloadPlugin  # noqa: E402


def _check_compatibility(manifest: PluginManifest, plugin_name: str) -> None:
    spec = SpecifierSet(manifest.requires_mate_bench)
    if __version__ not in spec:
        raise RuntimeError(
            f"Plugin '{plugin_name}' requires mate-bench {manifest.requires_mate_bench}, "
            f"but {__version__} is installed. Update the plugin or mate-bench."
        )


def load_workloads() -> dict[str, WorkloadPlugin]:
    plugins: dict[str, WorkloadPlugin] = {}
    for ep in entry_points(group="mate_bench.workload"):
        instance = ep.load()()
        _check_compatibility(instance.manifest, ep.name)
        plugins[instance.name] = instance
    return plugins


def load_engines() -> dict[str, EnginePlugin]:
    plugins: dict[str, EnginePlugin] = {}
    for ep in entry_points(group="mate_bench.engine"):
        instance = ep.load()()
        _check_compatibility(instance.manifest, ep.name)
        plugins[instance.name] = instance
    return plugins


def load_runtimes() -> dict[str, RuntimePlugin]:
    plugins: dict[str, RuntimePlugin] = {}
    for ep in entry_points(group="mate_bench.runtime"):
        instance = ep.load()()
        _check_compatibility(instance.manifest, ep.name)
        plugins[instance.name] = instance
    return plugins
