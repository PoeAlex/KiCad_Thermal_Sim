#!/usr/bin/env python
"""
Test runner for ThermalSim.

This script ensures all mocks are installed before running tests.
Run with: python run_tests.py [pytest arguments]

Example:
    python run_tests.py -v
    python run_tests.py -v tests/unit/test_stackup_parser.py
    python run_tests.py -v -m physics
    python run_tests.py --cov=. --cov-report=html
"""

import sys
import os

# Change to plugin directory
plugin_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(plugin_dir)

# Add tests directory to path first
sys.path.insert(0, os.path.join(plugin_dir, 'tests'))

# Install mocks BEFORE any other imports
from mocks.pcbnew_mock import install_mock
from mocks.wx_mock import install_wx_mock
install_mock()
install_wx_mock()

# Check for required dependencies
missing = []
try:
    import numpy
except ImportError:
    missing.append('numpy')

try:
    import scipy
except ImportError:
    missing.append('scipy')

try:
    import matplotlib
except ImportError:
    missing.append('matplotlib')

try:
    import pytest
except ImportError:
    missing.append('pytest')

if missing:
    print("Missing required dependencies:")
    for dep in missing:
        print(f"  - {dep}")
    print("\nInstall with:")
    print("  pip install " + " ".join(missing))
    sys.exit(1)

# Add plugin directory to path for package imports
# This allows importing ThermalSim as a package
parent_dir = os.path.dirname(plugin_dir)
sys.path.insert(0, parent_dir)

# Run pytest with any provided arguments
if __name__ == '__main__':
    args = sys.argv[1:] if len(sys.argv) > 1 else ['-v', 'tests/']
    sys.exit(pytest.main(args))
