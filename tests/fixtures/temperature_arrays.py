"""
Temperature array generators for testing.

This module provides functions to create temperature arrays
for testing visualization and solver functions.
"""

import numpy as np


def create_uniform_temperature(
    layers: int,
    rows: int,
    cols: int,
    temperature: float = 25.0
) -> np.ndarray:
    """
    Create a uniform temperature array.

    Parameters
    ----------
    layers : int
        Number of layers.
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    temperature : float
        Uniform temperature value.

    Returns
    -------
    np.ndarray
        Temperature array of shape (layers, rows, cols).
    """
    return np.ones((layers, rows, cols)) * temperature


def create_gradient_temperature(
    layers: int,
    rows: int,
    cols: int,
    t_min: float = 25.0,
    t_max: float = 100.0,
    direction: str = "vertical"
) -> np.ndarray:
    """
    Create a temperature array with a linear gradient.

    Parameters
    ----------
    layers : int
        Number of layers.
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    t_min : float
        Minimum temperature.
    t_max : float
        Maximum temperature.
    direction : str
        Gradient direction: "vertical", "horizontal", or "layer".

    Returns
    -------
    np.ndarray
        Temperature array of shape (layers, rows, cols).
    """
    T = np.zeros((layers, rows, cols))

    if direction == "vertical":
        grad = np.linspace(t_max, t_min, rows)
        for l in range(layers):
            for r in range(rows):
                T[l, r, :] = grad[r]

    elif direction == "horizontal":
        grad = np.linspace(t_min, t_max, cols)
        for l in range(layers):
            T[l, :, :] = grad[np.newaxis, :]

    elif direction == "layer":
        grad = np.linspace(t_max, t_min, layers)
        for l in range(layers):
            T[l, :, :] = grad[l]

    else:
        raise ValueError(f"Unknown direction: {direction}")

    return T


def create_hotspot_temperature(
    layers: int,
    rows: int,
    cols: int,
    ambient: float = 25.0,
    hotspot_temp: float = 100.0,
    hotspot_row: int = None,
    hotspot_col: int = None,
    hotspot_radius: int = 5
) -> np.ndarray:
    """
    Create a temperature array with a circular hotspot.

    Parameters
    ----------
    layers : int
        Number of layers.
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    ambient : float
        Ambient temperature.
    hotspot_temp : float
        Maximum hotspot temperature.
    hotspot_row : int, optional
        Hotspot center row (default: center).
    hotspot_col : int, optional
        Hotspot center column (default: center).
    hotspot_radius : int
        Hotspot radius in pixels.

    Returns
    -------
    np.ndarray
        Temperature array of shape (layers, rows, cols).
    """
    T = np.ones((layers, rows, cols)) * ambient

    if hotspot_row is None:
        hotspot_row = rows // 2
    if hotspot_col is None:
        hotspot_col = cols // 2

    y, x = np.ogrid[:rows, :cols]
    dist_sq = (y - hotspot_row)**2 + (x - hotspot_col)**2
    radius_sq = hotspot_radius**2

    # Gaussian falloff from hotspot center
    hotspot_mask = dist_sq <= radius_sq * 4  # Include tail
    hotspot_values = ambient + (hotspot_temp - ambient) * np.exp(-dist_sq / (2 * radius_sq))

    # Apply to top layer most strongly, decreasing with depth
    for l in range(layers):
        decay = 1.0 - l / (layers + 1)  # Top layer gets full heat
        layer_hotspot = ambient + (hotspot_values - ambient) * decay
        T[l] = np.where(hotspot_mask, np.maximum(T[l], layer_hotspot), T[l])

    return T


def create_multilayer_test_temperature(
    layer_count: int = 4,
    grid_size: int = 50,
    ambient: float = 25.0,
    max_temp: float = 80.0
) -> np.ndarray:
    """
    Create a test temperature array for multi-layer visualization.

    This creates a temperature distribution that decreases from top
    to bottom layer, simulating heat flow through the PCB.

    Parameters
    ----------
    layer_count : int
        Number of layers.
    grid_size : int
        Grid dimension (rows = cols = grid_size).
    ambient : float
        Ambient temperature.
    max_temp : float
        Maximum temperature at top layer hotspot.

    Returns
    -------
    np.ndarray
        Temperature array of shape (layer_count, grid_size, grid_size).
    """
    T = create_hotspot_temperature(
        layers=layer_count,
        rows=grid_size,
        cols=grid_size,
        ambient=ambient,
        hotspot_temp=max_temp,
        hotspot_radius=grid_size // 10
    )
    return T


def create_heatsink_mask(
    rows: int,
    cols: int,
    region: tuple = None
) -> np.ndarray:
    """
    Create a heatsink mask array.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    region : tuple, optional
        Region to mark as heatsink: (row_start, row_end, col_start, col_end).
        If None, marks bottom quarter of the grid.

    Returns
    -------
    np.ndarray
        Heatsink mask of shape (rows, cols) with 1.0 where heatsink present.
    """
    H = np.zeros((rows, cols))

    if region is None:
        # Default: bottom quarter, center half
        rs = 3 * rows // 4
        re = rows
        cs = cols // 4
        ce = 3 * cols // 4
    else:
        rs, re, cs, ce = region

    H[rs:re, cs:ce] = 1.0
    return H
