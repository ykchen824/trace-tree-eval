"""Base protocol for trajectory adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from trace_tree_eval.schema import Trajectory


class TraceAdapter(Protocol):
    """Protocol implemented by all trace/dataset adapters."""

    def load(self, source: str | Path | list[dict[str, Any]] | dict[str, Any]) -> list[Trajectory]:
        """Load one or more trajectories from a file path, HF repo, or parsed dict(s)."""
        ...
