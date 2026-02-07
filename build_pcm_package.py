"""
Build a PCM (Plugin and Content Manager) compatible ZIP for ThermalSim.

Usage:
    python build_pcm_package.py
    python build_pcm_package.py --version 0.2.0
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

VERSION = "0.1.0"

IDENTIFIER = "com.github.poealex.kicad-thermal-sim"

# Plugin source files to include (relative to ThermalSim/)
PLUGIN_FILES = [
    "__init__.py",
    "thermal_plugin.py",
    "capabilities.py",
    "stackup_parser.py",
    "gui_dialogs.py",
    "geometry_mapper.py",
    "thermal_solver.py",
    "visualization.py",
    "thermal_report.py",
    "pwl_parser.py",
    "dependency_installer.py",
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
            "2.5D transient thermal simulation for multilayer PCBs. "
            "Simulates heat spreading using finite volume methods with "
            "BDF2 time integration, directly within KiCad."
        ),
        "description_full": (
            "ThermalSim performs 2.5D transient thermal simulation of "
            "multilayer PCBs. Select heat-source pads, configure simulation "
            "parameters, and visualize temperature distribution across all "
            "copper layers.\n\n"
            "Features:\n"
            "- Automatic stackup detection from board file\n"
            "- Multi-layer heat spreading with via coupling\n"
            "- BDF2 implicit time integration\n"
            "- Piecewise-linear (PWL) time-varying power profiles\n"
            "- Heatsink/thermal-pad support (User.Eco1 layer)\n"
            "- HTML report with embedded thermal images\n\n"
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
            arc_name = f"plugins/ThermalSim/{filename}"
            zf.write(src_path, arc_name)

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
