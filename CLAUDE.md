# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ThermalSim is a KiCad PCB Editor plugin for 2.5D transient thermal simulation. It simulates heat spreading across multilayer PCBs directly within KiCad, using finite volume methods with BDF2 time integration.

## Dependencies

The plugin requires packages installed in KiCad's Python environment:
```bash
# Run in "KiCad 9.0 Command Prompt"
pip install matplotlib scipy
```

Optional: `pypardiso` (Intel MKL solver), `numba` (JIT compilation)

## Architecture

**Modular implementation** (8 modules):

```
ThermalSim/
├── __init__.py           # Plugin registration
├── capabilities.py       # Feature detection (HAS_LIBS, HAS_PARDISO, HAS_NUMBA)
├── stackup_parser.py     # S-expression parser for .kicad_pcb files
├── gui_dialogs.py        # SettingsDialog (wx.Dialog)
├── geometry_mapper.py    # PCB-to-grid mapping (FillContext, create_multilayer_maps)
├── thermal_solver.py     # Matrix assembly, BDF2 solver (SolverConfig, SolverResult)
├── visualization.py      # Matplotlib plotting functions
├── thermal_report.py     # HTML report generation
└── thermal_plugin.py     # Controller/orchestrator (ThermalPlugin class)
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `capabilities.py` | Runtime detection of numpy, matplotlib, pypardiso, numba |
| `stackup_parser.py` | Parse copper/dielectric layers from .kicad_pcb S-expressions |
| `gui_dialogs.py` | wxPython dialog for simulation parameters |
| `geometry_mapper.py` | Convert PCB geometry to discretized conductivity arrays |
| `thermal_solver.py` | Sparse matrix assembly, BDF2 time integration |
| `visualization.py` | Generate thermal plots and preview images |
| `thermal_report.py` | Generate HTML summary report |
| `thermal_plugin.py` | Orchestrate workflow, KiCad ActionPlugin interface |

### Key Data Structures

- `FillContext` (geometry_mapper.py): Holds K, V, H arrays and grid parameters
- `SolverConfig` (thermal_solver.py): Simulation configuration (time, steps, etc.)
- `SolverResult` (thermal_solver.py): Results including T array and diagnostics
- `K[layers, rows, cols]`: Relative thermal conductivity map (k_FR4=1.0, k_Cu~400)
- `V_map[rows, cols]`: Via enhancement factors for vertical coupling
- `H_map[rows, cols]`: Heatsink/thermal-pad mask (User.Eco1 layer)

## Physical Constants

| Constant | Value | Module |
|----------|-------|--------|
| k_Cu | 390 W/(m-K) | thermal_plugin.py, thermal_solver.py |
| k_FR4 | 0.3 W/(m-K) | thermal_plugin.py, thermal_solver.py |
| h_convection | 10 W/(m^2-K) | thermal_solver.py |
| via_factor | 1300 (k_Cu/k_FR4) | thermal_plugin.py, geometry_mapper.py |

## Testing

### Unit Tests (pytest)

Run tests using the batch file (handles mock injection):
```bash
run_tests.bat                    # All tests
run_tests.bat -k "test_solver"   # Single test file
run_tests.bat -m physics         # Physics validation tests only
run_tests.bat --cov=ThermalSim   # With coverage report
```

Test structure:
```
tests/
├── conftest.py           # Mock injection, fixtures
├── mocks/
│   ├── pcbnew_mock.py    # Full KiCad pcbnew mock
│   └── wx_mock.py        # wxPython mock
├── fixtures/
│   ├── sample_boards.py  # .kicad_pcb content generators
│   ├── stackup_configs.py
│   └── temperature_arrays.py
└── unit/
    ├── test_thermal_solver.py   # Physics validation (critical)
    ├── test_stackup_parser.py   # S-expression parsing
    ├── test_geometry_mapper.py  # Grid mapping
    ├── test_visualization.py    # PNG generation
    ├── test_thermal_report.py   # HTML report
    ├── test_gui_dialogs.py      # Settings parsing
    └── test_capabilities.py     # Feature detection
```

Key physics tests in `test_thermal_solver.py`:
- `test_steady_state_energy_balance` - Pin ≈ Pout at equilibrium
- `test_bdf2_convergence_order` - O(dt²) error reduction
- `test_monotonic_temperature_decrease_from_source` - Physical validity

### Manual Testing in KiCad

1. Open a PCB in KiCad PCB Editor
2. Select pads as heat sources
3. Run plugin via Tools -> External Plugins -> 2.5D Thermal Sim
4. Use "Preview" button for geometry verification before "Run"

Debug output goes to KiCad's Scripting Console (View -> Scripting Console):
```python
print(f"[ThermalSim] debug info here")
```

## Key Numerical Methods

- **Sparse matrix assembly**: COO format converted to CSR (scipy.sparse)
- **Time integration**: BDF2 (Backward Differentiation Formula, 2nd order)
- **Linear solver**: scipy.sparse.linalg.splu (SuperLU) or optional pypardiso
- **Multi-phase time stepping**: Phase A (8%, dt*0.5), Phase B (35%, dt*1.0), Phase C (57%, dt*2.0)

## pcbnew API Usage

The plugin uses KiCad's Python API extensively:
- `pcbnew.GetBoard()` - Active board
- `board.Footprints()` / `fp.Pads()` - Component pads
- `board.Tracks()` - Traces and vias (check `"VIA" in str(type(t)).upper()`)
- `board.Zones()` - Copper pours
- `zone.HitTestFilledArea(layer_id, pos, margin)` - Pixel-accurate zone detection (KiCad 9)

## Output Files

Generated in timestamped subfolder (e.g., `Thermalsim_20260131_143022/`):
- `thermal_report.html` - Summary with embedded images
- `thermal_preview.png` - Geometry visualization
- `thermal_stackup.png` or `thermal_final.png` - Temperature results
- `snap_*.png` - Time-series snapshots (if enabled)
- `thermal_sim_last_settings.json` - Persisted user settings (in plugin root)

## Module Dependencies

```
capabilities.py (no internal dependencies)
     |
stackup_parser.py (no internal dependencies)
     |
gui_dialogs.py (no internal dependencies)
     |
geometry_mapper.py (no internal dependencies)
     |
thermal_solver.py --> capabilities.py
     |
visualization.py (no internal dependencies)
     |
thermal_report.py (no internal dependencies)
     |
thermal_plugin.py --> all modules
     |
__init__.py --> thermal_plugin.py
```

## Docstring Standard

All public functions use NumPy-style docstrings:
- Summary (1 line)
- Extended description (optional)
- Parameters with types
- Returns with types
- Raises (if relevant)
- Examples (for complex functions)
