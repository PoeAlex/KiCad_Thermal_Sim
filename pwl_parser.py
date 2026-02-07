"""
Piecewise Linear (PWL) power profile parser.

This module parses LTspice-style PWL files that define time-varying
power profiles for thermal simulation heat sources.

File format:
    - Two columns: time (seconds), power (watts)
    - Whitespace-separated (spaces or tabs)
    - Lines starting with ';' or '*' are comments
    - Blank lines are ignored
    - Time values must be strictly monotonically increasing
"""

from typing import List, Tuple

import numpy as np


def parse_pwl_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse an LTspice-style PWL file.

    Parameters
    ----------
    filepath : str
        Path to the PWL file.

    Returns
    -------
    times : np.ndarray
        Monotonically increasing time values in seconds.
    powers : np.ndarray
        Power values in watts at each time point.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, has invalid format, or times are
        not strictly monotonically increasing.
    """
    times = []
    powers = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped[0] in (";", "*"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Line {line_num}: expected 2 columns, got {len(parts)}"
                )
            try:
                t = float(parts[0])
                p = float(parts[1])
            except ValueError:
                raise ValueError(
                    f"Line {line_num}: cannot parse '{parts[0]}' '{parts[1]}' as numbers"
                )
            times.append(t)
            powers.append(p)

    if not times:
        raise ValueError(f"PWL file '{filepath}' contains no data points")

    times_arr = np.array(times, dtype=np.float64)
    powers_arr = np.array(powers, dtype=np.float64)

    # Validate monotonicity
    if len(times_arr) > 1:
        diffs = np.diff(times_arr)
        if np.any(diffs <= 0):
            bad_idx = int(np.argmax(diffs <= 0))
            raise ValueError(
                f"Times not strictly increasing at index {bad_idx + 1}: "
                f"t[{bad_idx}]={times_arr[bad_idx]}, t[{bad_idx + 1}]={times_arr[bad_idx + 1]}"
            )

    return times_arr, powers_arr


def interpolate_pwl(times: np.ndarray, powers: np.ndarray, t: float) -> float:
    """
    Linearly interpolate power at time t.

    Uses np.interp which clamps to the first/last value outside the
    defined time range.

    Parameters
    ----------
    times : np.ndarray
        Breakpoint times (monotonically increasing).
    powers : np.ndarray
        Power values at each breakpoint.
    t : float
        Time at which to evaluate.

    Returns
    -------
    float
        Interpolated power value in watts.
    """
    return float(np.interp(t, times, powers))


def validate_pwl(times: np.ndarray, powers: np.ndarray) -> List[str]:
    """
    Validate a parsed PWL profile and return warnings.

    Parameters
    ----------
    times : np.ndarray
        Breakpoint times.
    powers : np.ndarray
        Power values at each breakpoint.

    Returns
    -------
    list of str
        Warning messages (empty if no issues found).
    """
    warnings = []

    if np.any(powers < 0):
        neg_count = int(np.sum(powers < 0))
        warnings.append(f"Negative power values found ({neg_count} points)")

    if len(times) > 1:
        min_dt = float(np.min(np.diff(times)))
        if min_dt < 1e-9:
            warnings.append(f"Very short time segment: {min_dt:.2e} s")

    max_power = float(np.max(np.abs(powers)))
    if max_power > 1000.0:
        warnings.append(f"Very high power: {max_power:.1f} W")

    return warnings
