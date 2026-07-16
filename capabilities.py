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
HAS_NATIVE_CORE : bool
    True if the optional Windows x64 CPU DLL is available.
"""

import importlib.util
import os
import platform
import sys

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

if os.environ.get("THERMALSIM_HEADLESS", "").strip().lower() in {"1", "true", "yes", "on"}:
    # Importing wx in the same standalone process as a loaded pcbnew board
    # makes KiCad register its global image handlers twice. Detection is
    # sufficient for headless solver/benchmark use.
    HAS_WX = importlib.util.find_spec("wx") is not None
else:
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

try:
    from .native_core import HAS_NATIVE_CORE
except (ImportError, OSError):
    HAS_NATIVE_CORE = False


def is_pypardiso_supported_platform(platform_name=None, machine=None):
    """
    Return whether PyPardiso is supported on the current platform.

    Parameters
    ----------
    platform_name : str, optional
        Platform identifier. Defaults to ``sys.platform``.
    machine : str, optional
        CPU architecture. Defaults to ``platform.machine()``.

    Returns
    -------
    bool
        True for Windows/Linux on x86_64/amd64 architectures.
    """
    platform_id = (platform_name or sys.platform).lower()
    arch = (machine or platform.machine()).lower()
    return (
        (platform_id == "win32" or platform_id.startswith("linux"))
        and arch in ("x86_64", "amd64")
    )


def get_pypardiso_optional_dependency():
    """
    Return installer metadata for the optional PyPardiso accelerator.

    Returns
    -------
    dict
        Metadata used by the dependency installer to render the optional
        PyPardiso checkbox and decide whether it should be selected.
    """
    supported = is_pypardiso_supported_platform()
    enabled = supported and not HAS_PARDISO
    if HAS_PARDISO:
        status = "Already installed."
    elif supported:
        status = "Recommended accelerator for large simulations."
    else:
        status = "Unavailable on this platform; using SciPy solver fallback."
    return {
        "import_name": "pypardiso",
        "pip_name": "pypardiso",
        "label": "Install PyPardiso accelerator",
        "supported": supported,
        "installed": HAS_PARDISO,
        "enabled": enabled,
        "default_selected": enabled,
        "status": status,
    }


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
    lines.append(f"  Native matrix-free CPU core: {_avail(HAS_NATIVE_CORE)}")
    return "\n".join(lines)
