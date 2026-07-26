# KiCad Thermal Sim - Fast Multi-Layer Copper Thermal Simulation for KiCad

![Simulation results](docs/images/result.png "6-layer thermal simulation results")

**KiCad Thermal Sim** is a lightweight KiCad PCB Editor plugin for fast, layout-oriented **heat spreading simulation across all copper layers** (`F.Cu` to `B.Cu`, including inner layers).

This is **not** a full 3D CFD/FEA solver. It is intended as a practical engineering tool to quickly answer:

- Where are the **hotspots** on each copper layer?
- How much do **copper pours/planes** and **via stitching** help?
- Which layout variant is **better** in an A/B comparison?
- How does heat distribute through the **stackup**?
- How much extra heat is created by **DC current flow** in copper?
- Which grid settings were actually used when automatic coarsening protects runtime?

---

## What It Simulates

- **2D in-plane conduction** on each copper layer.
- **Vertical coupling** between adjacent copper layers through FR4 and via enhancement.
- **Manual power injection** from explicitly configured **Heat Sources**.
- Constant pad power and time-varying **PWL power profiles**.
- Optional **DC current-flow heating** from source/sink pads on KiCad nets.
- **Convection** to ambient on top and bottom outer surfaces.
- Optional **Thermal Pad** geometry on `User.Eco1` for stronger bottom-side heat removal.
- Current-path diagnostics: copper loss, effective resistance, equivalent voltage drop, grid quality, and a Joule loss map.

---

## Installation

### Option A: KiCad Plugin Manager

1. Download the latest `ThermalSim-vX.Y.Z.zip` from the [Releases](https://github.com/PoeAlex/KiCad_Thermal_Sim/releases) page.
2. In KiCad, open **Plugin and Content Manager**.
3. Click **Install from File...** and select the downloaded ZIP.
4. Restart KiCad.
5. On first run, the plugin detects missing dependencies and offers to install them automatically.

### Option B: Manual Copy

1. Download or clone this repository.
2. Copy the plugin folder, meaning the folder that contains `__init__.py`, `thermal_plugin.py`, and the other `.py` modules, into KiCad's plugin directory:

   - **Windows**: `%APPDATA%\kicad\9.0\scripting\plugins\`
   - **Linux**: `~/.local/share/kicad/9.0/scripting/plugins/`
   - **macOS**: `~/Library/Application Support/kicad/9.0/scripting/plugins/`

3. Restart KiCad.
4. On first run, the plugin offers to install missing packages (`numpy`, `scipy`, `matplotlib`) automatically.

Alternatively, install packages manually in the **KiCad 9.0 Command Prompt**:

```bash
pip install numpy scipy matplotlib
```

Optional for faster large solves on supported Windows/Linux x86_64 systems:

```bash
pip install pypardiso
```

Run the plugin in PCB Editor via **Tools -> External Plugins -> 2.5D Thermal Sim**.

---

## Quick Start

1. Open your PCB in **KiCad PCB Editor**.
2. Select one or more pads if you want to pre-fill manual heat sources.
3. Run **Tools -> External Plugins -> 2.5D Thermal Sim**.
4. Use **Heat Sources** to add pads that dissipate manual power, then enter constant W values or PWL file paths.
5. Set **Duration**, **Ambient**, and **Resolution** on the **Overview** tab.
6. Optionally use **Current Heating** to add source/sink pads and per-pad currents for copper `I^2R` heating.
7. Optionally use **Advanced** for the simulation area, thermal pad, convection, and solver settings.
8. Click **Preview** to check the mapped geometry, then **Run**.

![GUI overview tab](docs/images/gui_sim.png "Overview tab")

![GUI heat sources tab](docs/images/gui_powerpads.png "Heat Sources tab")

![GUI current heating tab](docs/images/gui_current-paths.png "Current Heating tab")

![GUI advanced tab](docs/images/gui_adv.png "Advanced tab")

---

## GUI Settings

The resizable dialog keeps board context and live preflight status visible above and below four compact tabs:

- **Overview** - duration, ambient temperature, target cell size, compute budget, live cost estimate, output, and collapsible board details.
- **Heat Sources** - manual heat sources selected independently from current terminals.
- **Current Heating** - DC current source/sink terminals for Joule heating.
- **Advanced** - collapsible simulation-area, thermal-pad, and solver settings.

The header summarizes the board, copper-layer count, heat sources, and current-balance state. The fixed footer shows the requested versus actual grid, readiness warnings, and the main **Preview** and **Run Simulation** actions. Help and settings import/export are available from **More**. After a successful run, the dialog retains maximum temperature, elapsed time, and shortcuts to the report and output folder.

### Overview Tab

#### Board Info

Shows detected copper layers with thicknesses and dielectric gaps parsed from the board stackup. Pads selected when the dialog opened are shown only as a convenience; simulation roles are configured in **Heat Sources** and **Current Heating**.

#### Main Settings

- **Duration (sec)** - total simulated time. Shorter durations emphasize transient peaks; longer durations approach quasi steady-state.
- **Ambient Temp (C)** - reference temperature. Results are relative to ambient.
- **Target Cell Size (mm)** - desired spatial grid size. Smaller values improve hotspot and trace localization but increase runtime.
- **Compute Budget** - choose **Fast**, **Balanced**, **Detailed**, **Very detailed**, or **Custom**. The budget is based on total solver nodes across all copper layers.
- **Maximum Solver Nodes** - available in Custom mode for an explicit upper
  limit of up to 100 million equivalent uniform nodes. Values above 10 million
  are expert settings and may require tens of GB of RAM before adaptive
  reduction.

The live estimate shows requested versus actual cell size, grid dimensions, total nodes, approximate resolvable feature size, memory range, and a relative runtime class. For large boards, ThermalSim may automatically coarsen the requested cell size to remain within the selected budget. The report records the requested and actual resolution, compute budget, and simulated board fraction.

#### Output

- **Show All Layers** - display results for all copper layers.
- **Save Snapshots** - store intermediate temperature images.
- **Snapshot Count** - number of intermediate snapshots.
- **Output Folder** - where the timestamped result folder is created.

#### Settings Files

- **Load Settings...** - load a JSON settings file from any folder and apply it to the open dialog.
- **Save Settings...** - save the current dialog values as a JSON settings file.

ThermalSim also keeps using `thermal_sim_last_settings.json` in the plugin folder for automatic last-used settings. Manual load/save files use the same JSON structure and can be stored per project or per experiment.

### Heat Sources Tab

Manual pad power is configured separately from current-flow terminals.

1. Select pads in KiCad.
2. Click **Add Selected**.
3. Enter a power value or PWL file path in **Power / PWL**.
4. Use **Apply** to write the value to selected table rows. If no row is selected, a comma-separated list is applied in table order.

The Heat Sources table is:

| Column | Meaning |
|--------|---------|
| Pad | Footprint/pad name |
| Net | KiCad net name |
| Layer | Pad layer |
| Power W/PWL | Constant W value or PWL file path |

Accepted power entries:

| Entry | Meaning |
|-------|---------|
| `1.0` | 1 W constant on selected/listed power pads |
| `1.0, 0.5, 2.0` | Per-pad constant power in table order |
| `C:\sim\ramp.pwl` | Same PWL profile for selected/listed pads |
| `1.0, C:\sim\ramp.pwl` | Pad 1 = 1 W constant, Pad 2 = PWL file |

For backward compatibility, pads selected before opening the dialog pre-fill the Heat Sources table when no saved `power_pads` setting exists.

#### PWL File Format

PWL files are LTspice-style text files with two columns:

```text
; Comment lines start with ; or *
; Time(s)  Power(W)
0.0        0.0
0.001      1.0
0.005      2.5
0.010      2.5
0.020      0.0
```

- Time is in seconds.
- Power is in watts.
- Time values must be monotonically increasing.
- Power is linearly interpolated between points and held at the first/last value outside the defined range.

### Current Heating Tab

Enable current heating to calculate copper losses from DC current flow.

1. Select source/sink pads in KiCad.
2. Add them to a current group.
3. Enter positive current for source pads and negative current for sink pads.
4. Make sure every active KiCad net balances to `0 A`.

Current heating is additive with **Heat Sources**:

```text
total heat = manual pad power + calculated Joule copper loss
```

Power pads and current terminals may be different pads. Area limiting remains available with current heating: ThermalSim includes every pad, track, via/PTH, and filled zone belonging to the active current nets, then expands that geometry by the configured thermal margin. A large active plane may therefore require almost the full board.

The report shows **Path Current** as the useful current value. For a `+5 A` source and a `-5 A` sink, the path current is `5 A`. The internal sum of absolute terminal currents (`10 A` in that example) is kept only in raw diagnostics.

### Advanced Tab

#### Geometry Filters

- **Ignore Traces** - exclude copper traces from the conductivity map; zones, pours, and pads still contribute.
- **Limit to Active Sources and Current Paths** - crop the rectangular solver domain around heat sources and complete active current-net geometry.
- **Thermal Margin (mm)** - additional board area around sources and current paths. A practical starting point is 10-30 mm.

The preflight status shows the effective dimensions and percentage of the board. All copper inside the rectangular domain remains part of the thermal model. If active-net geometry cannot be inspected safely, ThermalSim falls back to the full board and reports a warning.

#### Thermal Pad (`User.Eco1`)

- **Enable Pad Simulation** - treat `User.Eco1` geometry as a thermal interface zone with enhanced bottom-side heat removal.
- **Pad Thickness (mm)** - thermal interface thickness.
- **Pad Cond. (W/mK)** - thermal interface conductivity.
- **Pad Heat Cap. (J/m2K)** - additional areal heat capacity.

#### Solver

- **Convection h (W/m2K)** - convection coefficient for top/bottom surfaces. Default is 10.
- **PCB Thickness (mm)** - overall board thickness. Stackup thickness is used when available.
- **Compute Engine** - `Auto` selects the matrix-free CPU engine for large jobs and keeps the legacy sparse solver for small jobs. Both engines remain manually selectable.
- **Spatial Mesh** - `Adaptive` keeps the requested cell size at copper edges, pads, vias, sources, and thermal-pad boundaries while merging homogeneous interiors. `Uniform` keeps the full regular grid.
- **Adaptive Max Cell Ratio** - largest adaptive cell relative to the requested cell size. The default is 8.
- **Capabilities** - detected solver backend, including the optional `thermalsim_core.dll`.

The **Balanced** compute budget is the default. Presets use total solver nodes, so a four-layer board automatically receives fewer 2D cells than a two-layer board at the same budget. **Custom** exposes one explicit maximum-node value; ThermalSim retains headroom when it must auto-coarsen.

The fast engine applies the finite-volume operator without constructing the
large global COO/CSR matrix used by the legacy direct solver. Adaptive runs use
a conservative reduced graph and geometrically coarsened multigrid
preconditioning. Reports show the equivalent uniform nodes, actual adaptive
nodes, reduction factor, PCG iterations, and residual.

---

## Preview

The **Preview** button generates a geometry visualization showing copper distribution, power/current pad locations, via regions, and the effective simulation-area boundary on each layer. Its header reports the cropped dimensions and percentage of the board.

ThermalSim uses the copper-zone fills already stored by KiCad. If zones are
stale or unfilled, press **B** in PCB Editor before Preview or Run. The plugin
does not refill zones automatically because this can block the interface for
minutes on large boards and is unstable in some KiCad builds.

![Preview](docs/images/preview.png "KiCad editor with geometry preview")

---

## Report Output

Each run creates a timestamped result folder, for example `Thermalsim_20260426_111121/`.

Typical files:

- `thermal_report.html` - main report with settings, stackup, diagnostics, images, and interactive heatmap.
- `thermal_preview.png` - mapped geometry preview.
- `thermal_stackup.png` or `thermal_final.png` - final temperature heatmap.
- `joule_loss_map.png` - current-induced copper loss map when current heating is active.
- `snap_*.png` - optional time-series snapshots.

### Current Path Diagnostics

When current heating is enabled, the HTML report includes **Current Path Diagnostics**:

- **Copper Loss** - total calculated Joule heating for active nets.
- **R_eff** - effective resistance, computed from `P / I_source^2`.
- **V_eq** - equivalent voltage drop, computed from `P / I_source`.
- **Source/Sink Current** - source and sink totals per active net.
- **Path Current** - useful current value for the path, not the doubled absolute terminal sum.
- **Net Path Metrics** - terminals, balance, loss, resistance, voltage drop, cell count, edge count, via edges, and copper islands.
- **Current Terminals** - pad positions, matched grid cells, island IDs, and mean potentials.
- **Mapped KiCad Primitives** - pads, tracks, vias/PTHs, zones, track length, and width summary.
- **Joule Loss Map** - per-layer static image of current-induced copper loss.

The report records requested and actual grid resolution, grid size, compute-budget preset, node budget, and the simulated percentage of the board.

---

## How to Interpret Results

This tool is most reliable for:

- **Relative comparisons** between layout variants.
- **Hotspot locations**.
- Trends from more copper, more vias, wider traces, or better spreading.
- Order-of-magnitude current-path loss and resistance diagnostics.

Absolute temperatures are **estimates** and depend strongly on modeling assumptions, grid resolution, board environment, airflow, and component/package thermal paths.

---

## Architecture

The plugin is split into focused modules:

| Module | Purpose |
|--------|---------|
| `capabilities.py` | Runtime detection of numpy, scipy, matplotlib, pypardiso, numba |
| `dependency_installer.py` | Auto-install dialog for missing packages via pip |
| `stackup_parser.py` | Parse copper/dielectric layers from `.kicad_pcb` S-expressions |
| `gui_dialogs.py` | wxPython dialog for simulation parameters |
| `geometry_mapper.py` | Accurately rasterize tracks, arcs, pads, vias, and filled zones |
| `adaptive_mesh.py` | Build conservative adaptive meshes and multigrid operators |
| `electrical_solver.py` | Solve DC current flow and convert copper losses to heat sources |
| `thermal_solver.py` | Legacy sparse and matrix-free PCG/BDF2 thermal solvers |
| `native_core.py` | Load the optional Windows x64 C++/OpenMP CPU core |
| `pwl_parser.py` | Parse LTspice-style PWL power profiles |
| `visualization.py` | Generate thermal plots, previews, heatmap payloads, and Joule loss maps |
| `thermal_report.py` | Generate the HTML report |
| `thermal_plugin.py` | Orchestrate workflow and KiCad ActionPlugin integration |

---

## Limitations

- No explicit component/package thermal model; junction-to-case-to-pad behavior is not modeled.
- Convection is simplified as uniform top/bottom ambient coupling.
- Radiation is not modeled.
- Current-flow heating is static DC only; AC effects, skin effect, and temperature-dependent copper resistance are not modeled.
- Current-path accuracy depends on grid resolution; narrow traces and small pads may need a finer grid to match hand calculations closely.
- Very fine requested resolutions may be auto-coarsened when the selected compute budget is exceeded.
- Via coupling is an approximation.
- Results depend strongly on **Target Cell Size**, **Compute Budget**, and the selected simulation area and thermal margin.
- Thermal Pad (`User.Eco1`) is a simplification of real contact pressure, interface quality, and sink temperature.

---

## Suggested Workflow

1. Add manual dissipating components in **Heat Sources**.
2. Add source/sink terminals in **Current Heating** only when copper `I^2R` heating matters.
3. Press **B** in PCB Editor so every copper zone has a current fill.
4. Enable **Limit to Active Sources and Current Paths** and start with a moderate thermal margin, for example 20 mm.
5. Check the effective board percentage; active current nets are always included completely.
6. Tune **Target Cell Size** until hotspots and current-path metrics are stable. Try 0.5 mm, then 0.3 mm.
7. Increase the **Compute Budget** only after checking the live node, memory, and runtime estimate.
8. Save JSON settings for repeatable project comparisons.
9. Compare layout variants using the same settings.
10. Validate important designs with measurement or a full 3D thermal workflow.

---

## Testing

Run the test suite from the repository root:

```bat
run_tests.bat
```

Run the synthetic large-grid benchmark without a proprietary board file:

```bash
python benchmark_fast_engine.py --nodes 10000000 --layers 4 --solve-steps 3
```

Legacy matrix construction is automatically skipped above the configured
safety limit. The benchmark reports equivalent/adaptive node counts, operator
build times, solve time, iteration counts, and process memory.

To build the optional native Windows CPU core, install Visual Studio Build
Tools with the C++ workload, then run:

```bash
cmake -S native -B native/build
cmake --build native/build --config Release
```

The generated `native/bin/thermalsim_core.dll` is included automatically by
`build_pcm_package.py`. Without the DLL, the same fast-engine interface uses
the NumPy/SciPy CPU implementation.

Useful variants:

```bat
run_tests.bat -k "test_thermal_solver"
run_tests.bat -m physics
run_tests.bat --cov=ThermalSim
```

### Real KiCad Benchmark

Use KiCad's bundled Python to benchmark a real board without saving or modifying it:

```powershell
& "C:\Program Files\KiCad\9.0\bin\python.exe" benchmark_real_board.py `
  "C:\path\to\board.kicad_pcb" --scenario thermal --time-stepping auto
```

Set `THERMALSIM_HEADLESS=1` when importing ThermalSim as a library outside PCB Editor. This prevents ActionPlugin registration while keeping the solver and board readers available.

---

## License / Disclaimer

MIT License

This plugin provides engineering estimates intended for fast iteration and comparative analysis. For safety-critical or thermally constrained designs, validate with measurement and/or a full 3D thermal tool.
