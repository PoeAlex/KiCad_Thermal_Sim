"""Shared workflow models for responsive ThermalSim jobs."""

from dataclasses import dataclass, field
import hashlib
import json
import threading
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SimulationSettings:
    """Normalized settings used by preflight and execution."""

    requested_res_mm: float
    sim_time_s: float
    ambient_c: float
    output_dir: str
    current_enabled: bool = False
    limit_area: bool = False
    limit_distance_mm: float = 0.0
    backend: str = "auto"
    time_stepping: str = "auto"

    @classmethod
    def from_mapping(cls, values: Dict[str, Any]):
        """Create normalized settings from the legacy settings mapping."""
        return cls(
            requested_res_mm=float(values.get("res", 0.5)),
            sim_time_s=float(values.get("time", 20.0)),
            ambient_c=float(values.get("amb", 25.0)),
            output_dir=str(values.get("output_dir", "") or ""),
            current_enabled=bool(values.get("current_enabled", False)),
            limit_area=bool(values.get("limit_area", False)),
            limit_distance_mm=float(values.get("pad_dist_mm", 0.0) or 0.0),
            backend=str(values.get("solver_backend", "auto") or "auto").lower(),
            time_stepping=str(values.get("time_stepping", "auto") or "auto").lower(),
        )


@dataclass(frozen=True)
class BoardSnapshot:
    """Lightweight immutable board identity captured on the KiCad thread."""

    filename: str
    fingerprint: str
    bbox_mm: Tuple[float, float, float, float]
    copper_layers: Tuple[int, ...]
    track_count: int
    footprint_count: int
    zone_count: int


@dataclass(frozen=True)
class GridEstimate:
    """Final solver grid after area limiting and automatic coarsening."""

    requested_res_mm: float
    actual_res_mm: float
    x_min_mm: float
    y_min_mm: float
    width_mm: float
    height_mm: float
    rows: int
    cols: int
    layer_count: int
    auto_coarsened: bool
    expert_limits: bool
    max_cells: int
    target_cells: int

    @property
    def base_cells(self):
        return self.rows * self.cols

    @property
    def nodes(self):
        return self.base_cells * self.layer_count

    @property
    def complexity(self):
        if self.nodes < 150_000:
            return "Low"
        if self.nodes < 500_000:
            return "Medium"
        return "High"


@dataclass
class PreflightResult:
    """Structured validation result rendered by the settings dialog."""

    grid: Optional[GridEstimate] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ready(self):
        return not self.errors

    @property
    def status(self):
        if self.errors:
            return "Blocked"
        if self.warnings:
            return "Warning"
        return "Ready"


@dataclass(frozen=True)
class JobProgress:
    """Progress message passed from the simulation pipeline to the GUI."""

    stage: str
    current: int
    total: int
    message: str = ""


class CancellationToken:
    """Thread-safe cooperative cancellation signal."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()


@dataclass
class SimulationArtifacts:
    """Files produced by a completed or failed run."""

    report_path: Optional[str] = None
    preview_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    run_dir: Optional[str] = None
    status: str = "running"


class GeometryCache:
    """Single-entry geometry cache optimized for Preview -> Run reuse."""

    def __init__(self):
        self.key = None
        self.value = None

    def get(self, key):
        return self.value if key == self.key else None

    def put(self, key, value):
        self.key = key
        self.value = value

    def clear(self):
        self.key = None
        self.value = None


def stable_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible data."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def geometry_cache_key(snapshot: BoardSnapshot, grid: GridEstimate, settings, pad_keys):
    """Build a cache key containing only geometry-affecting inputs."""
    payload = {
        "board": snapshot.fingerprint,
        "grid": {
            "res": grid.actual_res_mm,
            "rows": grid.rows,
            "cols": grid.cols,
            "x": grid.x_min_mm,
            "y": grid.y_min_mm,
        },
        "layers": snapshot.copper_layers,
        "ignore_traces": bool(settings.get("ignore_traces", False)),
        "ignore_polygons": bool(settings.get("ignore_polygons", False)),
        "limit_area": bool(settings.get("limit_area", False)),
        "pad_dist_mm": float(settings.get("pad_dist_mm", 0.0) or 0.0),
        "use_heatsink": bool(settings.get("use_heatsink", False)),
        "pads": sorted(str(key) for key in pad_keys),
    }
    return stable_fingerprint(payload)
