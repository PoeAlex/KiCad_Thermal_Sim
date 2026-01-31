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

**Single-file implementation** (`thermal_plugin.py`, ~2087 lines):

- **Lines 78-206**: `parse_stackup_from_board_file()` - Parses .kicad_pcb S-expression to extract copper layers and dielectric thicknesses
- **Lines 242-462**: `SettingsDialog` (wx.Dialog) - GUI for simulation parameters
- **Lines 464-2087**: `ThermalPlugin` (pcbnew.ActionPlugin) - Main plugin class
  - `RunSafe()` (503-1276): Core simulation loop including matrix assembly and BDF2 solver
  - `create_multilayer_maps()` (1277-1500): Extracts copper geometry from PCB into conductivity arrays
  - `_write_html_report()` (1906-2087): Generates HTML output with embedded images

**Key data structures:**
- `K[layers, rows, cols]`: Relative thermal conductivity map (k_FR4=1.0, k_Cu≈400)
- `V_map[rows, cols]`: Via enhancement factors for vertical coupling
- `H_map[rows, cols]`: Heatsink/thermal-pad mask (User.Eco1 layer)

## Physical Constants (hardcoded)

| Constant | Value | Location |
|----------|-------|----------|
| k_Cu | 390 W/(m·K) | Line 729 |
| k_FR4 | 0.3 W/(m·K) | Line 730 |
| h_convection | 10 W/(m²·K) | Lines 885, 891 |
| via_factor | 1300 (k_Cu/k_FR4) | Line 696 |

## Testing

No automated tests exist. Manual testing workflow:
1. Open a PCB in KiCad PCB Editor
2. Select pads as heat sources
3. Run plugin via Tools → External Plugins → 2.5D Thermal Sim
4. Use "Preview" button for geometry verification before "Run"

Debug output goes to KiCad's Scripting Console (View → Scripting Console):
```python
print(f"[ThermalSim] debug info here")
```

## Key Numerical Methods

- **Sparse matrix assembly**: COO format converted to CSR (scipy.sparse)
- **Time integration**: BDF2 (Backward Differentiation Formula, 2nd order)
- **Linear solver**: scipy.sparse.linalg.splu (SuperLU) or optional pypardiso
- **Multi-phase time stepping**: Phase A (8%, dt×0.5), Phase B (35%, dt×1.0), Phase C (57%, dt×2.0)

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
