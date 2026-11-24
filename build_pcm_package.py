"""
Build a PCM (Plugin and Content Manager) compatible ZIP for ThermalSim.

Usage:
    python build_pcm_package.py
    python build_pcm_package.py --version 0.2.3
    python build_pcm_package.py --output-dir dist

The output ZIP can be installed in KiCad via:
    Plugin and Content Manager -> Install from File
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile

VERSION = "0.4.6"

IDENTIFIER = "com.github.poealex.kicad-thermal-sim"

# Plugin source files to include (relative to ThermalSim/)
PLUGIN_FILES = [
    "__init__.py",
    "__main__.py",
    "simulation_core.py",
    "thermal_plugin.py",
    "headless_api.py",
    "thermalsim_cli.py",
    "capabilities.py",
    "native_core.py",
    "stackup_parser.py",
    "gui_dialogs.py",
    "geometry_mapper.py",
    "adaptive_mesh.py",
    "electrical_solver.py",
    "thermal_solver.py",
    "visualization.py",
    "thermal_report.py",
    "pwl_parser.py",
    "dependency_installer.py",
    "workflow.py",
    "ThermalSim_icon.png",
]

SCHEMA_FILES = [
    "schema/job.schema.json",
    "schema/summary.schema.json",
]


def build_metadata(version):
    """
    Build PCM metadata.json content.

    Parameters
    ----------
    version : str
        Semantic version string (e.g. "0.1.0").

    Returns
    -------
    dict
        Metadata dictionary conforming to KiCad PCM schema v1.
    """
    return {
        "$schema": "https://go.kicad.org/pcm/schemas/v1",
        "name": "ThermalSim - 2.5D Thermal Simulation",
        "description": (
            "Fast 2.5D steady-state and transient thermal simulation for "
            "multilayer PCBs in KiCad 9 & 10, powered by a Native C++/OpenMP "
            "multigrid CPU solver."
        ),
        "description_full": (
            "ThermalSim performs fast 2.5D steady-state and transient thermal "
            "simulation of multilayer PCBs in KiCad 9 & 10. Select heat-source "
            "pads, configure simulation parameters, and visualize temperature "
            "distribution across all copper layers.\n\n"
            "Features:\n"
            "- KiCad 9.0 & 10.0 compatibility\n"
            "- Direct Steady-State mode (single-solve final temperature) and "
            "BDF2 transient time integration with early steady-state stop\n"
            "- Native C++/OpenMP v4 Compact Low-RAM Multigrid & Fused PCG CPU "
            "core (Windows, Linux, macOS) with mixed-precision V-cycles and "
            "automatic NumPy/SciPy fallback\n"
            "- Automatic stackup detection from board file\n"
            "- Multi-layer heat spreading with via coupling\n"
            "- DC current-flow Joule heating in traces, vias, and zones\n"
            "- Auto, Uniform, and Adaptive spatial meshing up to 100M nodes\n"
            "- Piecewise-linear (PWL) time-varying power profiles\n"
            "- Heatsink/thermal-pad support (User.Eco1 layer) with fixed sink "
            "or dynamic 0D heatsink (R_th + thermal mass)\n"
            "- Compressed .npz geometry caching and HTML report with "
            "hotspot-preserving interactive heatmaps\n"
            "- Headless CLI & JSON API (inspect, validate, run, sweep, compare) "
            "with semantic summary.json for AI agents and CI automation\n\n"
            "Requires: numpy, scipy, matplotlib (auto-install dialog included)"
        ),
        "identifier": IDENTIFIER,
        "type": "plugin",
        "author": {
            "name": "Alex Poe",
            "contact": {
                "github": "https://github.com/poealex"
            }
        },
        "license": "MIT",
        "resources": {
            "homepage": "https://github.com/PoeAlex/KiCad_Thermal_Sim"
        },
        "versions": [
            {
                "version": version,
                "status": "stable",
                "kicad_version": "9.0",
            }
        ],
    }


def build_zip(version, output_dir):
    """
    Build the PCM ZIP package.

    Parameters
    ----------
    version : str
        Semantic version string.
    output_dir : str
        Directory to write the ZIP file to.

    Returns
    -------
    str
        Path to the created ZIP file.
    """
    # Locate ThermalSim source directory (same dir as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)
    zip_name = f"ThermalSim-v{version}.zip"
    zip_path = os.path.join(output_dir, zip_name)

    metadata = build_metadata(version)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Write metadata.json at root
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        # Write plugin source files
        for filename in PLUGIN_FILES:
            src_path = os.path.join(script_dir, filename)
            if not os.path.isfile(src_path):
                print(f"WARNING: {filename} not found, skipping", file=sys.stderr)
                continue
            arc_name = f"plugins/{filename}"
            zf.write(src_path, arc_name)

        # Write JSON schemas when present
        for schema_rel in SCHEMA_FILES:
            schema_src = os.path.join(script_dir, schema_rel)
            if os.path.isfile(schema_src):
                zf.write(schema_src, f"plugins/{schema_rel}")

        for binary_name in (
            "thermalsim_core.dll",
            "libthermalsim_core.so",
            "libthermalsim_core.dylib",
        ):
            native_bin = os.path.join(
                script_dir, "native", "bin", binary_name
            )
            if os.path.isfile(native_bin):
                zf.write(native_bin, f"plugins/{binary_name}")

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Build ThermalSim PCM package")
    parser.add_argument("--version", default=VERSION,
                        help=f"Package version (default: {VERSION})")
    parser.add_argument("--output-dir", default="dist",
                        help="Output directory (default: dist)")
    args = parser.parse_args()

    zip_path = build_zip(args.version, args.output_dir)

    # Compute SHA-256
    sha256 = hashlib.sha256(open(zip_path, 'rb').read()).hexdigest()

    print(f"Built: {zip_path}")
    print(f"SHA-256: {sha256}")

    # List contents
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print(f"\nContents ({len(zf.namelist())} files):")
        for name in sorted(zf.namelist()):
            info = zf.getinfo(name)
            print(f"  {name}  ({info.file_size} bytes)")


if __name__ == "__main__":
    main()
