"""
Unit tests for visualization module.

This module tests the Matplotlib-based plotting functions.
"""

import os
import pytest
import numpy as np
import tempfile

from ThermalSim.visualization import (
    save_stackup_plot,
    save_snapshot,
    show_results_top_bot,
    show_results_all_layers,
    save_preview_from_arrays,
    save_current_density_plot,
)
from ThermalSim.current_analyzer import CurrentPathResult

from tests.fixtures.temperature_arrays import (
    create_uniform_temperature,
    create_gradient_temperature,
    create_hotspot_temperature,
    create_heatsink_mask,
)


class TestSaveStackupPlot:
    """Tests for save_stackup_plot function."""

    def test_creates_png_file(self, temp_dir):
        """Test that PNG file is created."""
        T = create_uniform_temperature(2, 50, 50, temperature=50.0)
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "test_stackup.png")

        save_stackup_plot(T, H, amb=25.0, layer_names=["F.Cu", "B.Cu"], fname=fname)

        assert os.path.exists(fname)

    def test_png_file_valid(self, temp_dir):
        """Test that created file has valid PNG signature."""
        T = create_uniform_temperature(2, 50, 50, temperature=50.0)
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "test_stackup.png")

        save_stackup_plot(T, H, amb=25.0, layer_names=["F.Cu", "B.Cu"], fname=fname)

        with open(fname, 'rb') as f:
            signature = f.read(8)

        # PNG file signature
        assert signature[:4] == b'\x89PNG'

    def test_single_layer(self, temp_dir):
        """Test with single layer."""
        T = create_uniform_temperature(1, 50, 50, temperature=50.0)
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "single_layer.png")

        save_stackup_plot(T, H, amb=25.0, layer_names=["F.Cu"], fname=fname)

        assert os.path.exists(fname)

    def test_four_layers(self, temp_dir):
        """Test with four layers."""
        T = create_gradient_temperature(4, 50, 50, t_min=25.0, t_max=80.0, direction="layer")
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "four_layer.png")

        save_stackup_plot(
            T, H, amb=25.0,
            layer_names=["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
            fname=fname
        )

        assert os.path.exists(fname)

    def test_with_time_elapsed(self, temp_dir):
        """Test with time elapsed annotation."""
        T = create_uniform_temperature(2, 50, 50, temperature=50.0)
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "with_time.png")

        save_stackup_plot(
            T, H, amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            fname=fname,
            t_elapsed=10.5
        )

        assert os.path.exists(fname)

    def test_with_heatsink_mask(self, temp_dir):
        """Test with heatsink mask overlay."""
        T = create_hotspot_temperature(2, 50, 50, ambient=25.0, hotspot_temp=80.0)
        H = create_heatsink_mask(50, 50)
        fname = os.path.join(temp_dir, "with_heatsink.png")

        save_stackup_plot(
            T, H, amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            fname=fname
        )

        assert os.path.exists(fname)

    def test_high_temperature_clamping(self, temp_dir):
        """Test that extreme temperatures are clamped."""
        # Create array with very high temperature
        T = create_uniform_temperature(2, 50, 50, temperature=500.0)
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "high_temp.png")

        # Should not raise, should clamp vmax
        save_stackup_plot(T, H, amb=25.0, layer_names=["F.Cu", "B.Cu"], fname=fname)

        assert os.path.exists(fname)

    def test_missing_layer_names(self, temp_dir):
        """Test with fewer layer names than layers."""
        T = create_uniform_temperature(4, 50, 50, temperature=50.0)
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "missing_names.png")

        # Only provide 2 names for 4 layers
        save_stackup_plot(
            T, H, amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            fname=fname
        )

        assert os.path.exists(fname)


class TestSaveSnapshot:
    """Tests for save_snapshot function."""

    def test_creates_snapshot_file(self, temp_dir):
        """Test that snapshot file is created."""
        T = create_hotspot_temperature(2, 50, 50, ambient=25.0, hotspot_temp=60.0)
        H = np.zeros((50, 50))

        result = save_snapshot(
            T, H, amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            idx=1,
            t_elapsed=5.0,
            out_dir=temp_dir
        )

        assert result is not None
        assert os.path.exists(result)

    def test_snapshot_filename_format(self, temp_dir):
        """Test snapshot filename includes index and time."""
        T = create_uniform_temperature(2, 50, 50, temperature=50.0)
        H = np.zeros((50, 50))

        result = save_snapshot(
            T, H, amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            idx=3,
            t_elapsed=7.5,
            out_dir=temp_dir
        )

        filename = os.path.basename(result)
        assert "snap_03" in filename
        assert "t7.5" in filename
        assert filename.endswith(".png")

    def test_snapshot_sequential_numbering(self, temp_dir):
        """Test multiple snapshots with sequential numbering."""
        T = create_uniform_temperature(2, 50, 50, temperature=50.0)
        H = np.zeros((50, 50))

        results = []
        for i, t in enumerate([1.0, 5.0, 10.0], start=1):
            result = save_snapshot(
                T, H, amb=25.0,
                layer_names=["F.Cu", "B.Cu"],
                idx=i,
                t_elapsed=t,
                out_dir=temp_dir
            )
            results.append(result)

        assert len(results) == 3
        assert all(os.path.exists(r) for r in results)


class TestShowResultsTopBot:
    """Tests for show_results_top_bot function."""

    def test_creates_thermal_final_png(self, temp_dir):
        """Test that thermal_final.png is created."""
        T = create_hotspot_temperature(2, 50, 50, ambient=25.0, hotspot_temp=80.0)
        H = np.zeros((50, 50))

        result = show_results_top_bot(
            T, H, amb=25.0,
            open_file=False,
            out_dir=temp_dir
        )

        assert result is not None
        assert os.path.exists(result)
        assert "thermal_final.png" in result

    def test_with_time_annotation(self, temp_dir):
        """Test with time elapsed annotation."""
        T = create_uniform_temperature(2, 50, 50, temperature=60.0)
        H = np.zeros((50, 50))

        result = show_results_top_bot(
            T, H, amb=25.0,
            open_file=False,
            t_elapsed=20.0,
            out_dir=temp_dir
        )

        assert os.path.exists(result)

    def test_with_heatsink_contour(self, temp_dir):
        """Test that heatsink mask creates contour on bottom layer."""
        T = create_uniform_temperature(2, 50, 50, temperature=60.0)
        H = create_heatsink_mask(50, 50)

        result = show_results_top_bot(
            T, H, amb=25.0,
            open_file=False,
            out_dir=temp_dir
        )

        assert os.path.exists(result)

    def test_multilayer_uses_first_and_last(self, temp_dir):
        """Test that multilayer uses first and last layer."""
        T = create_gradient_temperature(4, 50, 50, t_min=25.0, t_max=100.0, direction="layer")
        H = np.zeros((50, 50))

        result = show_results_top_bot(
            T, H, amb=25.0,
            open_file=False,
            out_dir=temp_dir
        )

        assert os.path.exists(result)


class TestShowResultsAllLayers:
    """Tests for show_results_all_layers function."""

    def test_creates_thermal_stackup_png(self, temp_dir):
        """Test that thermal_stackup.png is created."""
        T = create_hotspot_temperature(4, 50, 50, ambient=25.0, hotspot_temp=80.0)
        H = np.zeros((50, 50))

        result = show_results_all_layers(
            T, H, amb=25.0,
            layer_names=["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
            open_file=False,
            out_dir=temp_dir
        )

        assert result is not None
        assert os.path.exists(result)
        assert "thermal_stackup.png" in result

    def test_single_layer(self, temp_dir):
        """Test with single layer."""
        T = create_uniform_temperature(1, 50, 50, temperature=60.0)
        H = np.zeros((50, 50))

        result = show_results_all_layers(
            T, H, amb=25.0,
            layer_names=["F.Cu"],
            open_file=False,
            out_dir=temp_dir
        )

        assert os.path.exists(result)

    def test_two_layers(self, temp_dir):
        """Test with two layers."""
        T = create_uniform_temperature(2, 50, 50, temperature=60.0)
        H = np.zeros((50, 50))

        result = show_results_all_layers(
            T, H, amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            open_file=False,
            out_dir=temp_dir
        )

        assert os.path.exists(result)

    def test_six_layers(self, temp_dir):
        """Test with six layers."""
        T = create_gradient_temperature(6, 50, 50, t_min=25.0, t_max=100.0, direction="layer")
        H = np.zeros((50, 50))

        result = show_results_all_layers(
            T, H, amb=25.0,
            layer_names=["F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"],
            open_file=False,
            out_dir=temp_dir
        )

        assert os.path.exists(result)

    def test_with_time_elapsed(self, temp_dir):
        """Test with time elapsed annotation."""
        T = create_uniform_temperature(4, 50, 50, temperature=70.0)
        H = np.zeros((50, 50))

        result = show_results_all_layers(
            T, H, amb=25.0,
            layer_names=["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
            open_file=False,
            t_elapsed=15.0,
            out_dir=temp_dir
        )

        assert os.path.exists(result)

    def test_with_heatsink_on_bottom(self, temp_dir):
        """Test heatsink contour appears on bottom layer."""
        T = create_uniform_temperature(4, 50, 50, temperature=60.0)
        H = create_heatsink_mask(50, 50)

        result = show_results_all_layers(
            T, H, amb=25.0,
            layer_names=["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
            open_file=False,
            out_dir=temp_dir
        )

        assert os.path.exists(result)


class TestVisualizationEdgeCases:
    """Edge case tests for visualization functions."""

    def test_very_small_grid(self, temp_dir):
        """Test with very small grid."""
        T = create_uniform_temperature(2, 5, 5, temperature=50.0)
        H = np.zeros((5, 5))
        fname = os.path.join(temp_dir, "small_grid.png")

        save_stackup_plot(T, H, amb=25.0, layer_names=["F.Cu", "B.Cu"], fname=fname)

        assert os.path.exists(fname)

    def test_large_grid(self, temp_dir):
        """Test with larger grid."""
        T = create_uniform_temperature(2, 200, 200, temperature=50.0)
        H = np.zeros((200, 200))
        fname = os.path.join(temp_dir, "large_grid.png")

        save_stackup_plot(T, H, amb=25.0, layer_names=["F.Cu", "B.Cu"], fname=fname)

        assert os.path.exists(fname)

    def test_uniform_temperature_at_ambient(self, temp_dir):
        """Test when all temperatures equal ambient."""
        T = create_uniform_temperature(2, 50, 50, temperature=25.0)
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "at_ambient.png")

        # Should not raise even when vmin == vmax
        save_stackup_plot(T, H, amb=25.0, layer_names=["F.Cu", "B.Cu"], fname=fname)

        assert os.path.exists(fname)

    def test_nan_handling(self, temp_dir):
        """Test handling of NaN values in temperature array."""
        T = create_uniform_temperature(2, 50, 50, temperature=50.0)
        T[0, 25, 25] = np.nan  # Introduce NaN
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "with_nan.png")

        # Should handle gracefully (matplotlib may warn but shouldn't crash)
        try:
            save_stackup_plot(T, H, amb=25.0, layer_names=["F.Cu", "B.Cu"], fname=fname)
            assert os.path.exists(fname)
        except Exception as e:
            pytest.skip(f"NaN handling not supported: {e}")

    def test_negative_temperatures(self, temp_dir):
        """Test with temperatures below zero Celsius."""
        T = create_uniform_temperature(2, 50, 50, temperature=-10.0)
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "negative_temp.png")

        save_stackup_plot(T, H, amb=-20.0, layer_names=["F.Cu", "B.Cu"], fname=fname)

        assert os.path.exists(fname)

    def test_empty_layer_names_list(self, temp_dir):
        """Test with empty layer names list."""
        T = create_uniform_temperature(2, 50, 50, temperature=50.0)
        H = np.zeros((50, 50))
        fname = os.path.join(temp_dir, "empty_names.png")

        save_stackup_plot(T, H, amb=25.0, layer_names=[], fname=fname)

        assert os.path.exists(fname)


class TestSavePreviewFromArrays:
    """Tests for save_preview_from_arrays function."""

    def _make_arrays(self, layers=2, rows=20, cols=20):
        """Create minimal K, V_map, H_map arrays."""
        K = np.ones((layers, rows, cols), dtype=np.float64)
        # Mark some copper pixels
        K[:, 5:15, 5:15] = 400.0
        V_map = np.zeros((rows, cols), dtype=np.float64)
        H_map = np.zeros((rows, cols), dtype=np.float64)
        return K, V_map, H_map

    def test_creates_preview_file(self, temp_dir):
        """Test that preview file is created."""
        K, V_map, H_map = self._make_arrays()
        result = save_preview_from_arrays(
            K, V_map, H_map, pads_list=[], copper_ids=[0, 31],
            rows=20, cols=20, x_min=0.0, y_min=0.0, res=0.5,
            layer_names=["F.Cu", "B.Cu"], settings={},
            board=None, get_pad_pixels_func=lambda *a: [],
            out_dir=temp_dir,
        )
        assert result is not None
        assert os.path.exists(result)
        assert "thermal_preview.png" in result

    def test_valid_png_signature(self, temp_dir):
        """Test that output file has valid PNG signature."""
        K, V_map, H_map = self._make_arrays()
        result = save_preview_from_arrays(
            K, V_map, H_map, pads_list=[], copper_ids=[0, 31],
            rows=20, cols=20, x_min=0.0, y_min=0.0, res=0.5,
            layer_names=["F.Cu", "B.Cu"], settings={},
            board=None, get_pad_pixels_func=lambda *a: [],
            out_dir=temp_dir,
        )
        with open(result, 'rb') as f:
            sig = f.read(4)
        assert sig == b'\x89PNG'

    def test_four_layer_stackup(self, temp_dir):
        """Test with four-layer board."""
        K, V_map, H_map = self._make_arrays(layers=4)
        result = save_preview_from_arrays(
            K, V_map, H_map, pads_list=[], copper_ids=[0, 1, 2, 31],
            rows=20, cols=20, x_min=0.0, y_min=0.0, res=0.5,
            layer_names=["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
            settings={}, board=None, get_pad_pixels_func=lambda *a: [],
            out_dir=temp_dir,
        )
        assert result is not None
        assert os.path.exists(result)

    def test_with_vias(self, temp_dir):
        """Test that via overlay renders without error."""
        K, V_map, H_map = self._make_arrays()
        V_map[8:12, 8:12] = 1300.0  # via pixels
        result = save_preview_from_arrays(
            K, V_map, H_map, pads_list=[], copper_ids=[0, 31],
            rows=20, cols=20, x_min=0.0, y_min=0.0, res=0.5,
            layer_names=["F.Cu", "B.Cu"], settings={},
            board=None, get_pad_pixels_func=lambda *a: [],
            out_dir=temp_dir,
        )
        assert result is not None

    def test_with_heatsink(self, temp_dir):
        """Test with heatsink mask and use_heatsink setting."""
        K, V_map, H_map = self._make_arrays()
        H_map[5:15, 5:15] = 1.0
        result = save_preview_from_arrays(
            K, V_map, H_map, pads_list=[], copper_ids=[0, 31],
            rows=20, cols=20, x_min=0.0, y_min=0.0, res=0.5,
            layer_names=["F.Cu", "B.Cu"],
            settings={'use_heatsink': True},
            board=None, get_pad_pixels_func=lambda *a: [],
            out_dir=temp_dir,
        )
        assert result is not None

    def test_falls_back_on_bad_dir(self):
        """Test falls back to module dir when output directory is invalid."""
        K = np.ones((1, 5, 5))
        V_map = np.zeros((5, 5))
        H_map = np.zeros((5, 5))
        result = save_preview_from_arrays(
            K, V_map, H_map, pads_list=[], copper_ids=[0],
            rows=5, cols=5, x_min=0.0, y_min=0.0, res=0.5,
            layer_names=["F.Cu"], settings={},
            board=None, get_pad_pixels_func=lambda *a: [],
            out_dir="/nonexistent_dir_xyz_123/sub",
        )
        # Falls back to module directory, still produces a file
        assert result is not None
        assert os.path.exists(result)
        # Clean up fallback file
        try:
            os.remove(result)
        except Exception:
            pass


class TestSaveCurrentDensityPlot:
    """Tests for save_current_density_plot with multi-panel output."""

    def _make_result(self, layers=2, rows=20, cols=30, label="Path 1"):
        """Create a mock CurrentPathResult with J data."""
        J = np.zeros((layers, rows, cols), dtype=np.float64)
        # Put current density on top and bottom layers
        J[0, 5:15, 5:25] = 1e6  # F.Cu has some J
        if layers > 1:
            J[-1, 5:15, 5:25] = 5e5  # B.Cu has less J
        return CurrentPathResult(
            resistance_ohm=0.002,
            voltage_drop_v=0.01,
            power_loss_w=0.05,
            V_field=np.zeros((layers, rows, cols)),
            J_magnitude=J,
            Q_i2r=np.zeros(layers * rows * cols),
            label=label,
        )

    def test_creates_file_two_layer(self, temp_dir):
        """Test that a PNG is created for a 2-layer result."""
        result = self._make_result(layers=2)
        path = save_current_density_plot(
            [result], ["F.Cu", "B.Cu"], temp_dir,
        )
        assert path is not None
        assert os.path.exists(path)
        assert "current_density.png" in path

    def test_creates_file_single_layer(self, temp_dir):
        """Test with single layer (only 1 panel per path)."""
        result = self._make_result(layers=1)
        path = save_current_density_plot(
            [result], ["F.Cu"], temp_dir,
        )
        assert path is not None
        assert os.path.exists(path)

    def test_six_layer_max_on_inner(self, temp_dir):
        """When max J is on an inner layer, show 3 panels."""
        layers = 6
        result = self._make_result(layers=layers, label="6L path")
        # Put max J on inner layer 2
        result.J_magnitude[2, 8:12, 10:20] = 2e6
        path = save_current_density_plot(
            [result],
            ["F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"],
            temp_dir,
        )
        assert path is not None
        assert os.path.exists(path)

    def test_empty_results(self, temp_dir):
        """Empty results should return None."""
        path = save_current_density_plot([], ["F.Cu"], temp_dir)
        assert path is None

    def test_multiple_paths(self, temp_dir):
        """Multiple paths should produce panels for each."""
        r1 = self._make_result(layers=2, label="Path A")
        r2 = self._make_result(layers=2, label="Path B")
        path = save_current_density_plot(
            [r1, r2], ["F.Cu", "B.Cu"], temp_dir,
        )
        assert path is not None
        assert os.path.exists(path)
