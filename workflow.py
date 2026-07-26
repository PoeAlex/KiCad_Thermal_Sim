"""Shared workflow models for responsive ThermalSim jobs."""

from dataclasses import dataclass, field
import ctypes
import hashlib
import json
import os
import pickle
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple


def get_process_memory_mb(peak=False):
    """Return current or peak process working set in MiB when supported."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.argtypes = []
            get_process.restype = ctypes.c_void_p
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(Counters),
                ctypes.c_ulong,
            ]
            get_memory.restype = ctypes.c_int
            if get_memory(get_process(), ctypes.byref(counters), counters.cb):
                value = (
                    counters.PeakWorkingSetSize
                    if peak else counters.WorkingSetSize
                )
                return float(value) / (1024.0 ** 2)
        except (AttributeError, OSError, ValueError):
            return None
    return None


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
    area_mode: str = "full"
    grid_detail_level: str = "balanced"
    grid_node_budget: int = 800000
    compute_engine: str = "auto"
    mesh_mode: str = "adaptive"
    adaptive_max_cell_ratio: int = 8
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
            area_mode=str(values.get("area_mode", "full") or "full"),
            grid_detail_level=str(
                values.get("grid_detail_level", "balanced") or "balanced"
            ),
            grid_node_budget=int(values.get("grid_node_budget", 800000) or 800000),
            compute_engine=str(
                values.get("compute_engine", "auto") or "auto"
            ).lower(),
            mesh_mode=str(values.get("mesh_mode", "adaptive") or "adaptive").lower(),
            adaptive_max_cell_ratio=int(
                values.get("adaptive_max_cell_ratio", 8) or 8
            ),
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
class AreaEstimate:
    """Effective rectangular simulation area selected from the PCB."""

    mode: str
    x_min_mm: float
    y_min_mm: float
    width_mm: float
    height_mm: float
    board_width_mm: float
    board_height_mm: float
    margin_mm: float = 0.0
    heat_source_count: int = 0
    active_net_names: Tuple[str, ...] = ()
    fallback_to_full: bool = False
    warnings: Tuple[str, ...] = ()

    @property
    def area_fraction(self):
        """Return the simulated fraction of the board bounding rectangle."""
        board_area = self.board_width_mm * self.board_height_mm
        if board_area <= 0.0:
            return 1.0
        return min(1.0, max(0.0, self.width_mm * self.height_mm / board_area))

    @property
    def limited(self):
        """Return whether the effective domain is smaller than the full board."""
        return self.mode != "full" and self.area_fraction < 0.999


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
    detail_level: str = "legacy"
    node_budget: int = 0
    memory_mb_low: int = 0
    memory_mb_high: int = 0
    runtime_class: str = "Unknown"

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

    @property
    def feature_min_mm(self):
        """Approximate smallest feature represented by at least two cells."""
        return self.actual_res_mm * 2.0

    @property
    def feature_max_mm(self):
        """Approximate feature size represented robustly by three cells."""
        return self.actual_res_mm * 3.0


@dataclass
class PreflightResult:
    """Structured validation result rendered by the settings dialog."""

    grid: Optional[GridEstimate] = None
    area: Optional[AreaEstimate] = None
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
    elapsed_s: Optional[float] = None
    max_temp_c: Optional[float] = None


class GeometryCache:
    """Geometry cache with optional persistent local-disk reuse."""

    def __init__(self, persistent=False, max_bytes=4 * 1024 ** 3, cache_dir=None):
        self.key = None
        self.value = None
        self.persistent = bool(persistent)
        self.max_bytes = max(0, int(max_bytes))
        default_root = (
            os.environ.get("LOCALAPPDATA")
            or os.path.join(os.path.expanduser("~"), ".cache")
        )
        self.cache_dir = cache_dir or os.path.join(
            default_root, "ThermalSim", "cache", "geometry-v1"
        )

    def _path(self, key):
        return os.path.join(self.cache_dir, f"{key}.pickle")

    def _prune(self, keep_path=None):
        if not self.persistent or self.max_bytes <= 0:
            return
        try:
            entries = [
                item for item in os.scandir(self.cache_dir)
                if item.is_file() and item.name.endswith(".pickle")
            ]
            total = sum(item.stat().st_size for item in entries)
            if total <= self.max_bytes:
                return
            entries.sort(key=lambda item: item.stat().st_mtime)
            for item in entries:
                if keep_path and os.path.normcase(item.path) == os.path.normcase(keep_path):
                    continue
                try:
                    size = item.stat().st_size
                    os.remove(item.path)
                    total -= size
                except OSError:
                    continue
                if total <= self.max_bytes:
                    break
        except OSError:
            return

    def get(self, key):
        if key == self.key:
            return self.value
        if not self.persistent:
            return None
        path = self._path(key)
        try:
            with open(path, "rb") as stream:
                value = pickle.load(stream)
            os.utime(path, None)
            self.key = key
            self.value = value
            return value
        except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
            try:
                os.remove(path)
            except OSError:
                pass
            return None

    def put(self, key, value):
        self.key = key
        self.value = value
        if not self.persistent:
            return
        path = self._path(key)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            handle, temp_path = tempfile.mkstemp(
                prefix=f"{key}.", suffix=".tmp", dir=self.cache_dir
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temp_path, path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            self._prune(keep_path=path)
        except (OSError, pickle.PickleError, AttributeError, TypeError):
            return

    def clear(self):
        self.key = None
        self.value = None


class ThermalOperatorCache:
    """Single-entry cache for assembled, source-independent thermal operators.

    The stiffness matrix and heat-capacity vector are unchanged when users
    adjust only power values, PWL files, snapshots, or report options.  Keeping
    this cache separate from geometry avoids reusing an operator after a
    material or boundary-condition change.
    """

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


class ThermalFactorizationCache:
    """Single-entry cache that releases native solver resources on eviction."""

    def __init__(self):
        self.key = None
        self.value = None

    def get(self, key):
        return self.value if key == self.key else None

    def put(self, key, value):
        if key != self.key:
            self.clear()
        self.key = key
        self.value = value

    def clear(self):
        if self.value is not None:
            release = getattr(self.value, "release", None)
            if callable(release):
                release()
        self.key = None
        self.value = None


def stable_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible data."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def geometry_cache_key(snapshot: BoardSnapshot, grid: GridEstimate, settings, pad_keys):
    """Build a cache key containing only geometry-affecting inputs."""
    payload = {
        "geometry_cache_version": 3,
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
