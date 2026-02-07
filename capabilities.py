"""
Feature detection for ThermalSim plugin.

This module provides runtime detection of optional dependencies and sets
feature flags that other modules can use to enable/disable functionality.

Attributes
----------
HAS_NUMPY : bool
    True if numpy is available.
HAS_SCIPY : bool
    True if scipy is available.
HAS_MATPLOTLIB : bool
    True if matplotlib is available.
HAS_WX : bool
    True if wxPython is available.
HAS_LIBS : bool
    True if all core dependencies (numpy, scipy, matplotlib, wx) are available.
HAS_PARDISO : bool
    True if pypardiso (Intel MKL sparse solver) is available.
HAS_NUMBA : bool
    True if numba (JIT compilation) is available.
"""

import importlib.util

# Granular detection of each core dependency
HAS_NUMPY = False
HAS_SCIPY = False
HAS_MATPLOTLIB = False
HAS_WX = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    pass

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    pass

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for file output
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    pass

try:
    import wx
    HAS_WX = True
except ImportError:
    pass

# Composite flag for backwards compatibility
HAS_LIBS = HAS_NUMPY and HAS_SCIPY and HAS_MATPLOTLIB and HAS_WX

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


def get_missing_packages():
    """
    Return a list of missing core packages as (import_name, pip_name) tuples.

    Returns
    -------
    list of tuple
        Each tuple contains (import_name, pip_name) for a missing package.
        Empty list if all core packages are installed.

    Examples
    --------
    >>> missing = get_missing_packages()
    >>> for imp_name, pip_name in missing:
    ...     print(f"pip install {pip_name}")
    """
    missing = []
    if not HAS_NUMPY:
        missing.append(("numpy", "numpy"))
    if not HAS_SCIPY:
        missing.append(("scipy", "scipy"))
    if not HAS_MATPLOTLIB:
        missing.append(("matplotlib", "matplotlib"))
    return missing


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
      numpy: Available
      scipy: Available
      matplotlib: Available
      wx: Available
      Core libs (all): Available
      PyPardiso (Intel MKL solver): Not available
      Numba (JIT compilation): Not available
    """
    _avail = lambda v: "Available" if v else "Not available"
    lines = ["ThermalSim Capabilities:"]
    lines.append(f"  numpy: {_avail(HAS_NUMPY)}")
    lines.append(f"  scipy: {_avail(HAS_SCIPY)}")
    lines.append(f"  matplotlib: {_avail(HAS_MATPLOTLIB)}")
    lines.append(f"  wx: {_avail(HAS_WX)}")
    lines.append(f"  Core libs (all): {_avail(HAS_LIBS)}")
    lines.append(f"  PyPardiso (Intel MKL solver): {_avail(HAS_PARDISO)}")
    lines.append(f"  Numba (JIT compilation): {_avail(HAS_NUMBA)}")
    return "\n".join(lines)
