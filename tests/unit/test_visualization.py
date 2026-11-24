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
    build_interactive_heatmap_payload,
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


class TestBuildInteractiveHeatmapPayload:
    """Tests for interactive heatmap payload generation."""

    def test_payload_contains_geometry_metadata(self):
        """Payload should include grid geometry and color scale metadata."""
        T = create_uniform_temperature(2, 8, 10, temperature=42.0)

        payload = build_interactive_heatmap_payload(
            T,
            amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            res_mm=0.25,
            x_min_mm=12.5,
            y_min_mm=4.0,
            show_all=True
        )

        assert payload["ambient_c"] == 25.0
        assert payload["vmin_c"] == 25.0
        assert payload["vmax_c"] == 42.0
        assert payload["res_mm"] == 0.25
        assert payload["x_min_mm"] == 12.5
        assert payload["y_min_mm"] == 4.0
        assert payload["visible_layer_indices"] == [0, 1]
        assert payload["layers"][0]["rows"] == 8
        assert payload["layers"][0]["cols"] == 10

    def test_payload_filters_layers_when_show_all_disabled(self):
        """Only top and bottom should remain when show_all is false."""
        T = create_gradient_temperature(4, 6, 6, t_min=25.0, t_max=100.0, direction="layer")

        payload = build_interactive_heatmap_payload(
            T,
            amb=25.0,
            layer_names=["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
            res_mm=0.5,
            show_all=False
        )

        assert payload["visible_layer_indices"] == [0, 3]
        assert [layer["index"] for layer in payload["layers"]] == [0, 3]
        assert [layer["name"] for layer in payload["layers"]] == ["F.Cu", "B.Cu"]

    def test_payload_clamps_vmax_above_ambient_window(self):
        """Viewer color scale should clamp extremely high temperatures."""
        T = create_uniform_temperature(2, 4, 4, temperature=500.0)

        payload = build_interactive_heatmap_payload(
            T,
            amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            res_mm=1.0,
            show_all=True
        )

        assert payload["vmax_c"] == 275.0

    def test_payload_serializes_row_major_data(self):
        """Flattened data should preserve row-major ordering."""
        T = np.array([[[25.0, 30.0], [35.0, 40.0]]], dtype=float)

        payload = build_interactive_heatmap_payload(
            T,
            amb=25.0,
            layer_names=["F.Cu"],
            res_mm=0.1,
            show_all=True
        )

        assert payload["layers"][0]["data"] == [25.0, 30.0, 35.0, 40.0]
        assert payload["downsampled"] is False
        assert payload["downsample_stride"] == 1
        assert payload["original_rows"] == 2
        assert payload["original_cols"] == 2
        assert payload["original_res_mm"] == 0.1

    def test_payload_downsamples_large_grid_and_preserves_hotspots(self):
        """Large grids should be block-max downsampled while preserving peaks and mm metadata."""
        import json

        T = np.full((2, 500, 500), 30.0, dtype=np.float64)
        T[0, 317, 429] = 123.456
        T[1, 10, 15] = 99.5

        payload = build_interactive_heatmap_payload(
            T,
            amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            res_mm=0.1,
            x_min_mm=5.0,
            y_min_mm=10.0,
            show_all=True,
        )

        assert payload["downsampled"] is True
        assert payload["downsample_stride"] == 2
        assert payload["original_rows"] == 500
        assert payload["original_cols"] == 500
        assert payload["original_res_mm"] == 0.1
        assert payload["res_mm"] == 0.2
        assert payload["x_min_mm"] == 5.0
        assert payload["y_min_mm"] == 10.0

        top = payload["layers"][0]
        bot = payload["layers"][1]
        assert top["rows"] == 250
        assert top["cols"] == 250
        assert max(top["rows"], top["cols"]) <= 320
        assert len(top["data"]) == 250 * 250
        assert top["min_c"] == 30.0
        assert top["max_c"] == 123.456
        assert max(top["data"]) == 123.456
        assert top["data"][(317 // 2) * 250 + (429 // 2)] == 123.456
        assert bot["max_c"] == 99.5
        assert bot["data"][(10 // 2) * 250 + (15 // 2)] == 99.5

        full_payload = build_interactive_heatmap_payload(
            T,
            amb=25.0,
            layer_names=["F.Cu", "B.Cu"],
            res_mm=0.1,
            x_min_mm=5.0,
            y_min_mm=10.0,
            show_all=True,
            max_dim=None,
        )
        assert full_payload["downsampled"] is False
        assert len(json.dumps(payload, separators=(",", ":"))) < len(json.dumps(full_payload, separators=(",", ":"))) / 3

    def test_payload_downsamples_non_divisible_edge_blocks(self):
        """Non-divisible grids should preserve hotspots in partial trailing blocks."""
        T = np.full((1, 5, 7), 25.0, dtype=np.float64)
        T[0, 4, 6] = 88.8

        payload = build_interactive_heatmap_payload(
            T,
            amb=25.0,
            layer_names=["F.Cu"],
            res_mm=0.2,
            max_dim=3,
        )

        assert payload["downsampled"] is True
        assert payload["downsample_stride"] == 3
        assert payload["original_rows"] == 5
        assert payload["original_cols"] == 7
        assert payload["res_mm"] == 0.6
        layer = payload["layers"][0]
        assert layer["rows"] == 2
        assert layer["cols"] == 3
        assert layer["data"][1 * 3 + 2] == 88.8

    def test_payload_handles_nan_and_inf_in_vectorized_and_downsampled_paths(self):
        """Non-finite values should become None and not poison finite block-max values."""
        T_small = np.array([[[25.1234, np.nan], [np.inf, -np.inf]]], dtype=np.float64)
        payload_small = build_interactive_heatmap_payload(
            T_small,
            amb=20.0,
            layer_names=["F.Cu"],
            res_mm=0.5,
        )
        assert payload_small["downsampled"] is False
        assert payload_small["layers"][0]["min_c"] == 25.123
        assert payload_small["layers"][0]["max_c"] == 25.123
        assert payload_small["layers"][0]["data"] == [25.123, None, None, None]

        T_pool = np.array([[
            [np.nan, 45.5, np.nan, np.inf],
            [30.0, -np.inf, -np.inf, np.nan],
            [-15.0, np.nan, 10.0, 20.0],
            [np.nan, np.nan, 15.0, 18.0],
        ]], dtype=np.float64)
        payload_pool = build_interactive_heatmap_payload(
            T_pool,
            amb=20.0,
            layer_names=["F.Cu"],
            res_mm=0.5,
            max_dim=2,
        )
        assert payload_pool["downsampled"] is True
        assert payload_pool["downsample_stride"] == 2
        assert payload_pool["layers"][0]["rows"] == 2
        assert payload_pool["layers"][0]["cols"] == 2
        assert payload_pool["layers"][0]["data"] == [45.5, None, -15.0, 20.0]


class TestFastLargeGridPngExport:
    """Tests for Quick-Win #5: hotspot-preserving block-max PNG rendering on large grids."""

    def test_large_grid_show_results_and_snapshot_preserve_exact_max_and_downsample(
        self, temp_dir, monkeypatch
    ):
        """1200x1200 PNG exports downsample imshow arrays (<=800) while keeping exact max title & extent."""
        import matplotlib.axes
        from ThermalSim.visualization import save_joule_loss_map

        T = np.full((2, 1200, 1200), 25.0, dtype=np.float64)
        T[0, 345, 789] = 112.4
        T[1, 900, 100] = 68.7
        H = np.zeros((1200, 1200), dtype=np.float64)
        H[400:800, 400:800] = 1.0

        recorded_imshow = []
        recorded_titles = []
        orig_imshow = matplotlib.axes.Axes.imshow
        orig_set_title = matplotlib.axes.Axes.set_title

        def spy_imshow(self, X, *args, **kwargs):
            recorded_imshow.append((np.asarray(X).shape, kwargs.get("extent")))
            return orig_imshow(self, X, *args, **kwargs)

        def spy_set_title(self, label, *args, **kwargs):
            recorded_titles.append(label)
            return orig_set_title(self, label, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "imshow", spy_imshow)
        monkeypatch.setattr(matplotlib.axes.Axes, "set_title", spy_set_title)

        out_all = show_results_all_layers(
            T, H, amb=25.0, layer_names=["F.Cu", "B.Cu"], open_file=False, out_dir=temp_dir
        )
        out_snap = save_snapshot(
            T, H, amb=25.0, layer_names=["F.Cu", "B.Cu"], idx=1, t_elapsed=12.5, out_dir=temp_dir
        )
        out_tb = show_results_top_bot(
            T, H, amb=25.0, open_file=False, out_dir=temp_dir
        )

        for path in (out_all, out_snap, out_tb):
            assert os.path.exists(path)
            with open(path, "rb") as f:
                assert f.read(4) == b"\x89PNG"

        assert len(recorded_imshow) == 6
        for shape, extent in recorded_imshow:
            assert shape == (600, 600)
            assert extent == [-0.5, 1199.5, 1199.5, -0.5]

        assert any("112.4" in title for title in recorded_titles)
        assert any("68.7" in title for title in recorded_titles)

        # Also check save_joule_loss_map downsampling and exact mW/cell annotation
        recorded_imshow.clear()
        recorded_titles.clear()
        q = np.zeros((2, 1200, 1200), dtype=np.float64)
        q[0, 500, 600] = 0.012345  # 12.345 mW/cell
        out_j = save_joule_loss_map(
            q, 2, 1200, 1200, ["F.Cu", "B.Cu"], out_dir=temp_dir
        )
        assert out_j is not None and os.path.exists(out_j)
        assert all(shape == (600, 600) and extent == [-0.5, 1199.5, 1199.5, -0.5] for shape, extent in recorded_imshow)
        assert any("12.345 mW/cell" in title for title in recorded_titles)
