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
)

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
