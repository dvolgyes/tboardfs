"""Attach analysis exports to a published run snapshot."""

from collections.abc import Callable
from functools import cached_property, partial
from typing import Any

from tboardfs.epoch_report import _EpochReport
from tboardfs.epochs import _Epochs
from tboardfs.run_catalog import _Catalog
from tboardfs.run_inventory import _Inventory


class _RunAnalysis:
    """Render analysis from canonical metadata only when a file is accessed."""

    _inventory: _Inventory
    _resolve: Callable[[], _Epochs]
    _dirs: dict[int, str]

    def __init__(
        self,
        inventory: _Inventory,
        resolve: Callable[[], _Epochs],
        dirs: dict[int, str],
    ) -> None:
        self._inventory = inventory
        self._resolve = resolve
        self._dirs = dirs

    @staticmethod
    def install(tree: dict[str, Any], resolve: Callable[[], _Epochs]) -> None:
        """Give both interfaces the same files and immutable run inputs."""
        children = tree["children"]
        breakdown = children.get("epoch-breakdown")
        dirs = (
            {}
            if breakdown is None
            else {
                int(name.removeprefix("epoch-")): f"epoch-breakdown/{name}/"
                for name in breakdown["children"]
            }
        )
        analysis = _RunAnalysis(_Inventory(tree), resolve, dirs)
        children["catalog.json"] = {
            "type": "analysis",
            "mtime": 0.0,
            "render": analysis.catalog,
        }
        if dirs:
            for fmt in ("json", "tsv"):
                children[f"epochs.{fmt}"] = {
                    "type": "analysis",
                    "mtime": 0.0,
                    "render": partial(analysis.epoch_table, fmt),
                }

    @cached_property
    def epochs(self) -> _Epochs:
        """Resolve markers once if readers request analysis of this snapshot."""
        return self._resolve()

    @cached_property
    def report(self) -> _EpochReport:
        """Keep one table for both epoch encodings."""
        return _EpochReport.collect(self._inventory, self.epochs, self._dirs)

    def catalog(self) -> bytes:
        """Render the run's canonical catalog."""
        return _Catalog.render(self._inventory, self.epochs)

    def epoch_table(self, fmt: str) -> bytes:
        """Render the scoped epoch coverage table."""
        return self.report.render(fmt)
