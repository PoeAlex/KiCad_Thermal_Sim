# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

ThermalSim (**v0.5.0**) is a KiCad 9.0 & 10.0 PCB Editor plugin and headless automation tool for fast 2.5D steady-state and transient thermal simulation. It simulates heat spreading across multilayer PCBs using finite volume methods, a Native C++/OpenMP **v4** Compact Low-RAM Multigrid & Fused PCG core (with pure NumPy/SciPy fallback), and BDF2 time integration.

## Dependencies

Core runtime packages required in KiCad's Python environment:
```bash
pip install numpy scipy matplotlib
```

Optional native acceleration (compiled shared library in `native/bin/`, packaged automatically into PCM releases):
- **Native C++/OpenMP v4 Core** (`thermalsim_core.dll` / `libthermalsim_core.so` / `libthermalsim_core.dylib`) built from `native/thermalsim_core.cpp`.

## Architecture

```
KiCad_Thermal_Sim/
├── __init__.py               # Plugin registration (and headless guard for CLI/library use)
├── __main__.py               # Module execution entrypoint (`python3 -m KiCad_Thermal_Sim`)
├── simulation_core.py        # Shared zero-wx board/grid/cost/stackup/power helpers
├── thermal_plugin.py         # KiCad ActionPlugin orchestrator, progress/ETA, and caches
├── headless_api.py           # Zero-GUI JSON API (inspect, validate, run, sweep, compare)
├── thermalsim_cli.py         # Agent-friendly JSON CLI entrypoint with semantic exit codes
├── capabilities.py           # Runtime detection of numpy, scipy, matplotlib, wx, native v1-v4 ABI
├── native_core.py            # ctypes loader and Python wrappers for Native C++/OpenMP v1-v4 kernels
├── native/
│   ├── CMakeLists.txt        # Cross-platform CMake build for thermalsim_core
│   └── thermalsim_core.cpp   # OpenMP v4 compact stencil, mixed-precision MG, & fused PCG kernels
├── stackup_parser.py         # S-expression parser for .kicad_pcb stackups
├── gui_dialogs.py            # wxPython tabbed SettingsDialog with live preflight & 0D heatsink UI
├── geometry_mapper.py        # Exact KiCad 9/10 pad/trace/via/zone/Eco1 rasterization & GeometryState
├── electrical_solver.py      # DC current-flow solver and copper Joule heating maps
├── thermal_solver.py         # Direct steady-state, FMG-PCG, BDF2, and 0D Sherman-Morrison solvers
├── workflow.py               # Compressed .npz geometry caching, operator caches, preflight dataclasses
├── pwl_parser.py             # LTspice-style piecewise-linear (PWL) power profile parser
├── visualization.py          # Fast PNG heatmaps, geometry previews, Joule maps, & HTML payload builder
├── thermal_report.py         # Standalone HTML report generator with interactive ROI heatmap
├── dependency_installer.py   # wxPython auto-installer dialog for missing pip dependencies
├── build_pcm_package.py      # KiCad PCM ZIP packager
└── schema/
    ├── job.schema.json       # JSON Schema Draft 2020-12 for headless job configs
    └── summary.schema.json   # JSON Schema Draft 2020-12 for run summary.json outputs
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `simulation_core.py` | Zero-`wx` single source of truth for board area/grid estimation, calibrated RAM/runtime cost estimation, stackup thickness derivation, capacity assembly, and multi-layer PTH/SMD sparse pad power distribution |
| `capabilities.py` | Runtime detection of `numpy`, `scipy`, `matplotlib`, `wx`, and native core ABI levels (`HAS_NATIVE_CORE`, `HAS_NATIVE_MG`, `HAS_NATIVE_COMPACT`, `HAS_NATIVE_PCG`) |
| `native_core.py` | Cross-platform loader (`ctypes`) and wrappers for the Native C++/OpenMP v4 kernels (`apply_compact_dot_native`, `pcg_update_xr_norm2_native`, `pcg_update_p_native`, `f64_to_f32_native`, `f32_to_f64_dot_native`, `dot_f64_native`) |
| `native/thermalsim_core.cpp` | C++/OpenMP v4 compact mask+LUT stencil operators (with exact harmonic series Cu/FR4 interface conductance and parallel FR4 on copper cells), `float32` multigrid smooth/restrict/prolong/coarsen kernels, and fused `float64` PCG vector/dot kernels |
| `stackup_parser.py` | Parse copper/dielectric layers and thicknesses from `.kicad_pcb` S-expressions |
| `gui_dialogs.py` | Resizable wxPython `SettingsDialog` (Overview, Heat Sources, Current Heating, Advanced) |
| `geometry_mapper.py` | Rasterize continuous sub-grid tracks, arcs, rotated/offset SMD/THT pads (`GetLayerSet()`), vias, fast scanline zone polygons, and `User.Eco1` thermal pads into `GeometryState` / `FillContext` |
| `electrical_solver.py` | Solve DC current flow per net (including arc-accurate `PCB_ARC` discretization and stackup `gap_m` coupling) and compute copper Joule losses & path diagnostics |
| `thermal_solver.py` | Direct steady-state, FMG-preconditioned PCG, mixed-precision multigrid, geometric matrix-free transient BDF2 (with PWL breakpoint/step-average integration and exact fixed-sink energy balance), and Dynamic 0D Heatsink (`_HeatsinkPhaseState` via Sherman-Morrison) |
| `workflow.py` | Safe compressed `.npz` disk `GeometryCache`, `ThermalOperatorCache`, `ThermalFactorizationCache`, `PreflightResult`, `CancellationToken` |
| `pwl_parser.py` | Parse LTspice-style time-varying PWL power profiles |
| `visualization.py` | Generate layer heatmaps, geometry previews, Joule loss maps, and hotspot-preserving downsampled interactive heatmap payloads |
| `thermal_report.py` | Generate self-contained HTML report with summary cards (including Heatsink Temp & Heat Removal Split) and interactive ROI viewer |
| `thermal_plugin.py` | KiCad `ActionPlugin` orchestrator with background thread offloading and throttled cancellable progress/ETA |
| `headless_api.py` | Zero-GUI Python API (`inspect_board`, `validate_job`, `run_headless_simulation`, `run_parameter_sweep`, `compare_runs`) |
| `thermalsim_cli.py` | JSON-over-stdout CLI entrypoint (`inspect`, `validate`, `run`, `sweep`, `compare`) |
| `__main__.py` | Package entrypoint for `python3 -m KiCad_Thermal_Sim` |
| `schema/` | JSON Schemas (`job.schema.json`, `summary.schema.json`) for headless automation |

### Key Data Structures

- `GeometryState` / `FillContext` (`geometry_mapper.py`): Compact copper/via/heatsink masks and grid parameters
- `CompactStructuredThermalOperator` / `StructuredThermalOperator` (`thermal_solver.py`): Matrix-free finite-volume operators evaluating Level-0 stencils on the fly from compact masks and per-layer LUTs
- `SolverConfig` / `SolverResult` (`thermal_solver.py`): Simulation configuration (steady-state/transient, heatsink, time stepping) and results/diagnostics
- `HeatsinkModel` (`thermal_solver.py`): Dynamic 0D heatsink parameters (`R_th`, `C_hs`, contact factor, sink mode)

## Physical Constants

| Constant | Value | Module |
|----------|-------|--------|
| `k_Cu` | 390 W/(m·K) | `thermal_plugin.py`, `thermal_solver.py`, `simulation_core.py` |
| `k_FR4` | 0.3 W/(m·K) | `thermal_plugin.py`, `thermal_solver.py`, `simulation_core.py` |
| `h_conv` / `h_conv_bottom` | 10 W/(m²·K) default (`>= 0.0`, `h_conv_bottom` defaults to `h_conv`) | `thermal_solver.py`, `gui_dialogs.py`, `headless_api.py` |
| `via_plating_um` / `via_fill` | `25.0` µm / `"plated"` default (`"plated"` \| `"copper"`), physical barrel conductance | `geometry_mapper.py`, `electrical_solver.py`, `gui_dialogs.py` |

## Building the Native C++/OpenMP v4 Core

### Linux (CMake or `g++`)
```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
# Or directly:
mkdir -p native/bin
g++ -O3 -fno-fast-math -fopenmp -shared -fPIC -std=c++17 native/thermalsim_core.cpp -o native/bin/libthermalsim_core.so
```

### Windows (CMake / MSVC)
```bash
cmake -S native -B native/build
cmake --build native/build --config Release
```

### macOS (CMake or `clang++`)
```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
```

Built binaries in `native/bin/` (`libthermalsim_core.so`, `thermalsim_core.dll`, `libthermalsim_core.dylib`) are gitignored and automatically bundled by `build_pcm_package.py`.

## Testing

### Unit Tests (`pytest` via `run_tests.py`)

`run_tests.py` installs `pcbnew_mock` and `wx_mock` before invoking `pytest`. Run with `uv` or your Python environment:

```bash
uv run --with pytest --with pytest-mock --with numpy --with scipy --with matplotlib python run_tests.py -v
uv run --with pytest --with pytest-mock --with numpy --with scipy --with matplotlib python run_tests.py -v tests/unit/test_thermal_solver.py
uv run --with pytest --with pytest-mock --with numpy --with scipy --with matplotlib python run_tests.py -m physics
```

On Windows:
```bat
run_tests.bat
run_tests.bat -k "test_thermal_solver"
```

### Test Layout

```
tests/
├── conftest.py
├── mocks/
│   ├── pcbnew_mock.py
│   └── wx_mock.py
├── fixtures/
│   ├── sample_boards.py
│   ├── stackup_configs.py
│   └── temperature_arrays.py
└── unit/
    ├── test_build_pcm_package.py
    ├── test_capabilities.py
    ├── test_cli.py
    ├── test_dependency_installer.py
    ├── test_electrical_solver.py
    ├── test_geometry_mapper.py
    ├── test_gui_dialogs.py
    ├── test_headless_api.py
    ├── test_native_core.py
    ├── test_pwl_parser.py
    ├── test_run_tests_batch.py
    ├── test_stackup_parser.py
    ├── test_thermal_plugin.py
    ├── test_thermal_report.py
    ├── test_thermal_solver.py
    ├── test_visualization.py
    └── test_workflow.py
```

### Headless CLI & Real KiCad Verification

Set `THERMALSIM_HEADLESS=1` when importing `KiCad_Thermal_Sim` outside KiCad PCB Editor (set automatically by `python3 -m KiCad_Thermal_Sim` and `thermalsim_cli.py`):

```bash
python3 -m KiCad_Thermal_Sim inspect board.kicad_pcb --pretty
python3 -m KiCad_Thermal_Sim validate board.kicad_pcb --pad U1.37=5 --steady-state --amb 25
python3 -m KiCad_Thermal_Sim run board.kicad_pcb --pad U1.37=5 --steady-state --amb 25 --use-heatsink --sink-mode heatsink --heatsink-rth 3.0 --heatsink-mass 100 --contact-factor 1.0 --out ./out_a
python3 -m KiCad_Thermal_Sim sweep board.kicad_pcb --param heatsink_rth_kw --values 1,3,5,10 --pad U1.37=5 --steady-state --out ./sweep_out
python3 -m KiCad_Thermal_Sim compare ./out_a ./out_b --pretty
```

Real-board and synthetic benchmarks:
```bash
python3 benchmark_fast_engine.py --nodes 6000000 --layers 4 --solve-steps 3
THERMALSIM_HEADLESS=1 python3 benchmark_real_board.py /path/to/board.kicad_pcb --scenario thermal --time-stepping auto
```

## Key Numerical Methods

- **Direct Steady-State (`run_steady_state` / `run_steady_state_matrix_free`)**: Solves $K \cdot T = Q + b$ directly in a single solve for constant heat sources (using `scipy.sparse.linalg.splu` for small grids or matrix-free PCG for large grids).
- **FMG-Preconditioned PCG**: Full-Multigrid (FMG) initial guess (`_fmg_initial_guess`) feeds a `float64` outer PCG solver (`_solve_pcg`) using Native C++/OpenMP v4 fused kernels (`apply_compact_dot_native`, `pcg_update_xr_norm2_native`, `pcg_update_p_native`, `f64_to_f32_native`, `f32_to_f64_dot_native`).
- **Mixed-Precision Multigrid**: Runs symmetric `float32` V-cycle smoothing (damped Jacobi), 2x2 restriction, and prolongation (`_CompactGeometricMultigrid` / `_StructuredGeometricMultigrid` / `_GeometricMultigrid`) inside the `float64` outer PCG loop, with coarse direct solve via `scipy.sparse.linalg.factorized`.
- **Geometric Matrix-Free Transient (`_build_geometric_phase_plan`)**: 5-phase geometrically growing BDF2 time stepping for `time_stepping="auto"` / `"geometric"` in matrix-free transient runs (reusing the cached MG hierarchy per phase and stopping losslessly when steady state is reached). For PWL profiles, solver steps align with PWL breakpoints and use exact step-averaged power quadrature with integrated energy balance diagnostics. Legacy direct solver supports `auto`, `multi_phase`, `two_phase`, and `uniform` BDF2.
- **Exact In-Plane Sheet Conductance & Energy Balance**: Copper/FR4 interface conductances use exact series harmonic mean of effective cell sheet conductances (including parallel FR4 on copper cells), and fixed-temperature thermal-pad energy balance evaluates heat flow against `pad_sink_temp_c`.
- **Dynamic 0D Heatsink via Sherman-Morrison (`_HeatsinkPhaseState`)**: Couples the lumped 0D heatsink node ($R_{\text{th}}$, $C_{\text{hs}}$) to the PCB `User.Eco1` thermal pad via an exact rank-1 Sherman-Morrison update $(A + \alpha u v^T)^{-1}$ using one precomputed solve per phase/steady solve.

## Output Files

Generated in a timestamped subfolder (e.g., `Thermalsim_20261124_183800/`) or `--out` directory:
- `thermal_report.html` — Self-contained HTML report with summary cards, static images, diagnostics, and interactive ROI heatmap
- `thermal_preview.png` — Mapped geometry preview
- `thermal_stackup.png` or `thermal_final.png` — Final temperature heatmap
- `joule_loss_map.png` — Current-induced copper loss map (when current heating is active)
- `current_density.png` — Per-layer DC current density map in A/mm² (when current heating is active)
- `snap_*.png` — Time-series snapshots (when enabled in transient mode)
- `summary.json` — Semantic JSON summary (written by `headless_api` / CLI)
- `thermal_sim_last_settings.json` — Persisted GUI settings (in plugin root)

## Docstring Standard

All public functions use NumPy-style docstrings (`Summary`, `Parameters`, `Returns`, `Raises`, `Examples`). Keep all code, comments, UI labels, and documentation 100% in English.
