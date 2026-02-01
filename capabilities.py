"""
Feature detection for ThermalSim plugin.

This module provides runtime detection of optional dependencies and sets
feature flags that other modules can use to enable/disable functionality.

Attributes
----------
HAS_LIBS : bool
    True if numpy, matplotlib, and wx are available.
HAS_PARDISO : bool
    True if pypardiso (Intel MKL sparse solver) is available.
HAS_NUMBA : bool
    True if numba (JIT compilation) is available.
"""

import importlib.util

# Core libraries required for plugin operation
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for file output
    import matplotlib.pyplot as plt
    import wx
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

# Optional: Intel MKL-based sparse solver (faster for large systems)
_pardiso_spec = importlib.util.find_spec("pypardiso")
if _pardiso_spec is not None:
    try:
        import pypardiso
        HAS_PARDISO = True
    except ImportError:
        HAS_PARDISO = False
else:
    HAS_PARDISO = False

# Optional: Numba JIT compilation (not currently used, reserved for future)
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def get_capabilities_summary():
    """
    Return a human-readable summary of detected capabilities.

    Returns
    -------
    str
        Multi-line string describing which optional features are available.

    Examples
    --------
    >>> print(get_capabilities_summary())
    ThermalSim Capabilities:
      Core libs (numpy, matplotlib, wx): Available
      PyPardiso (Intel MKL solver): Not available
      Numba (JIT compilation): Not available
    """
    lines = ["ThermalSim Capabilities:"]
    lines.append(f"  Core libs (numpy, matplotlib, wx): {'Available' if HAS_LIBS else 'Not available'}")
    lines.append(f"  PyPardiso (Intel MKL solver): {'Available' if HAS_PARDISO else 'Not available'}")
    lines.append(f"  Numba (JIT compilation): {'Available' if HAS_NUMBA else 'Not available'}")
    return "\n".join(lines)
