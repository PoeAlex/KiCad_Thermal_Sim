"""
Unit tests for thermal_report module.

This module tests the HTML report generation functionality.
"""

import json
import os
import re
import pytest
import tempfile

from ThermalSim.thermal_report import (
    _fmt,
    _esc,
    write_html_report,
)
from ThermalSim.visualization import build_interactive_heatmap_payload
from tests.fixtures.temperature_arrays import create_uniform_temperature


def _extract_embedded_heatmap_json(html_text):
    """Extract and parse the embedded interactive heatmap JSON payload."""
    match = re.search(
        r"<script id='interactive-heatmap-data' type='application/json'>(.*?)</script>",
        html_text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


class TestFmtHelper:
    """Tests for _fmt helper function."""

    def test_none_value(self):
        """Test formatting None value."""
        assert _fmt(None) == "n/a"

    def test_float_value(self):
        """Test formatting float with 4 decimal places."""
        assert _fmt(1.23456789) == "1.2346"
        assert _fmt(0.0001) == "0.0001"
        assert _fmt(100.0) == "100.0000"

    def test_integer_value(self):
        """Test formatting integer."""
        assert _fmt(42) == "42"
        assert _fmt(0) == "0"

    def test_string_value(self):
        """Test formatting string."""
        assert _fmt("test") == "test"

    def test_with_suffix(self):
        """Test formatting with suffix."""
        assert _fmt(1.5, " mm") == "1.5000 mm"
        assert _fmt(100, " W") == "100 W"
        assert _fmt(None, " mm") == "n/a"


class TestEscHelper:
    """Tests for _esc helper function."""

    def test_normal_text(self):
        """Test that normal text passes through."""
        assert _esc("Hello World") == "Hello World"

    def test_html_special_chars(self):
        """Test HTML special character escaping."""
        assert _esc("<script>") == "&lt;script&gt;"
        assert _esc("a & b") == "a &amp; b"
        assert _esc('"quoted"') == "&quot;quoted&quot;"
        assert _esc("it's") == "it&#x27;s"

    def test_none_value(self):
        """Test escaping None."""
        assert _esc(None) == ""

    def test_empty_string(self):
        """Test escaping empty string."""
        assert _esc("") == ""


class TestWriteHtmlReport:
    """Tests for write_html_report function."""

    @pytest.fixture
    def basic_report_params(self, temp_dir):
        """Create basic parameters for report generation."""
        interactive_heatmap = build_interactive_heatmap_payload(
            create_uniform_temperature(2, 6, 8, temperature=55.0),
            amb=25.0,
            layer_names=['F.Cu', 'B.Cu'],
            res_mm=0.5,
            x_min_mm=1.0,
            y_min_mm=2.0,
            show_all=True
        )
        return {
            'settings': {
                'power_str': '1.0',
                'time': 20.0,
                'amb': 25.0,
                'thick': 1.6,
                'res': 0.5,
                'show_all': True,
                'snapshots': False,
            },
            'stack_info': {
                'board_thickness_mm': 1.6,
                'copper': [
                    {'name': 'F.Cu', 'thickness_mm': 0.035},
                    {'name': 'B.Cu', 'thickness_mm': 0.035},
                ],
            },
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035, 0.035],
                'gap_mm_used': [1.53],
                'gap_fallback_used': False,
            },
            'pad_power': [
                ('U1:1', 0.5),
                ('U1:2', 0.5),
            ],
            'layer_names': ['F.Cu', 'B.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
            'interactive_heatmap': interactive_heatmap,
        }

    def test_report_file_created(self, basic_report_params):
        """Test that report file is created."""
        result = write_html_report(**basic_report_params)

        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".html")

    def test_report_contains_title(self, basic_report_params):
        """Test that report contains title."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "KiCad Thermal Sim Report" in content
        assert "Summary" in content

    def test_report_contains_settings(self, basic_report_params):
        """Test that report contains simulation settings."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "power_str" in content
        assert "1.0" in content
        assert "time" in content
        assert "20.0" in content

    def test_report_contains_pad_power(self, basic_report_params):
        """Test that report contains pad power information."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "U1:1" in content
        assert "U1:2" in content
        assert "0.5" in content

    def test_report_contains_layer_names(self, basic_report_params):
        """Test that report contains layer names."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "F.Cu" in content
        assert "B.Cu" in content

    def test_report_contains_thickness_info(self, basic_report_params):
        """Test that report contains thickness information."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "1.6" in content  # Board thickness
        assert "0.035" in content  # Copper thickness

    def test_report_with_preview_image(self, basic_report_params, temp_dir):
        """Test report with preview image path."""
        preview_path = os.path.join(temp_dir, "preview.png")
        # Create dummy file
        with open(preview_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG signature

        params = basic_report_params.copy()
        params['preview_path'] = preview_path

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "preview.png" in content
        assert "<img" in content
        assert "Open full-size image" in content
        assert "target='_blank'" in content

    def test_report_with_heatmap_image(self, basic_report_params, temp_dir):
        """Test report with heatmap image path."""
        heatmap_path = os.path.join(temp_dir, "heatmap.png")
        with open(heatmap_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')

        params = basic_report_params.copy()
        params['heatmap_path'] = heatmap_path

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "heatmap.png" in content
        assert "rel='noopener'" in content

    def test_report_without_images(self, basic_report_params):
        """Test report handles missing images gracefully."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "not available" in content.lower()
        assert "Geometry Preview" in content
        assert "Final Heatmap" in content

    def test_report_with_k_norm_info(self, basic_report_params):
        """Test report with k_norm_info debug information."""
        params = basic_report_params.copy()
        params['k_norm_info'] = {
            'strategy': 'implicit_fvm_bdf2',
            'backend': 'SciPy',
            'N': 800,
            'pin_w': 1.0,
            'pout_final_w': 0.98,
            't_fr4_eff_per_plane_mm': [0.765, 0.765],
        }

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "implicit_fvm_bdf2" in content
        assert "SciPy" in content

    def test_report_with_snapshots(self, basic_report_params, temp_dir):
        """Test report with snapshot files."""
        # Create dummy snapshot files
        snap1 = os.path.join(temp_dir, "snap_01_t1.0.png")
        snap2 = os.path.join(temp_dir, "snap_02_t5.0.png")
        for path in [snap1, snap2]:
            with open(path, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n')

        params = basic_report_params.copy()
        params['snapshot_files'] = [(1.0, snap1), (5.0, snap2)]

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "snap_01" in content
        assert "snap_02" in content
        assert "t = 1.0 s" in content
        assert "t = 5.0 s" in content

    def test_report_html_escaping(self, basic_report_params):
        """Test that special characters are properly escaped."""
        params = basic_report_params.copy()
        params['settings'] = {
            'test_key': '<script>alert("xss")</script>',
            'another': 'a & b',
        }

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should not contain unescaped script tags
        assert '<script>alert' not in content
        assert '&lt;script&gt;' in content

    def test_report_valid_html_structure(self, basic_report_params):
        """Test that report has valid HTML structure."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "</head>" in content
        assert "<body>" in content
        assert "</body>" in content

    def test_report_contains_interactive_viewer_markup(self, basic_report_params):
        """Interactive heatmap viewer should be embedded in report HTML."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "interactive-heatmap-data" in content
        assert "heatmap-canvas" in content
        assert "ROI Statistics" in content
        assert "Clear all" in content
        assert "roi-apply-all-layers" in content
        assert "Apply ROI to all layers" in content
        assert "type='checkbox' checked" in content

    def test_report_embeds_parseable_heatmap_payload_for_interactivity(self, basic_report_params):
        """Interactive viewer payload should remain valid JSON in the report HTML."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        payload = _extract_embedded_heatmap_json(content)

        assert payload["visible_layer_indices"] == [0, 1]
        assert [layer["name"] for layer in payload["layers"]] == ["F.Cu", "B.Cu"]
        assert payload["layers"][0]["rows"] == 6
        assert payload["layers"][0]["cols"] == 8
        assert "&quot;" not in content
        assert "buildOptions()" in content
        assert "sel.addEventListener('change'" in content
        assert "applyToAllLayers:roiToggle?roiToggle.checked:true" in content
        assert "roi.mode==='all_layers'" in content

    def test_report_contains_collapsible_debug_sections(self, basic_report_params):
        """Debug information should remain available in collapsed detail blocks."""
        params = basic_report_params.copy()
        params['k_norm_info'] = {'backend': 'SciPy', 'steps_total': 42}
        params['snapshot_debug'] = {'snapshots_enabled': True}

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "Diagnostics" in content
        assert "<details>" in content
        assert "<details open>" not in content
        assert "Solver Normalization and Debug" in content
        assert "Snapshot Debug" in content

    def test_report_contains_print_styles(self, basic_report_params):
        """Printable report CSS should be embedded."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "@media print" in content
        assert "viewer-controls" in content
        assert "interactive-heatmap-panel" in content
        assert "diagnostics-section" in content
        assert "print-hide-col" in content

    def test_report_snapshots_section_is_collapsible(self, basic_report_params, temp_dir):
        """Snapshots should be present behind a collapsible section."""
        snap1 = os.path.join(temp_dir, "snap_01_t1.0.png")
        with open(snap1, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')

        params = basic_report_params.copy()
        params['snapshot_files'] = [(1.0, snap1)]

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "Snapshots" in content
        assert "snapshot-card" in content
        assert "snap_01_t1.0.png" in content
        assert "Open snapshot" in content

    def test_report_4_layer_stackup(self, temp_dir):
        """Test report generation for 4-layer stackup."""
        params = {
            'settings': {'power_str': '1.0', 'time': 20.0},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035, 0.035, 0.035, 0.035],
                'gap_mm_used': [0.2, 1.0, 0.2],
                'gap_fallback_used': False,
            },
            'pad_power': [('U1:1', 1.0)],
            'layer_names': ['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
        }

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "F.Cu" in content
        assert "In1.Cu" in content
        assert "In2.Cu" in content
        assert "B.Cu" in content
        # Check gap interfaces
        assert "F.Cu -&gt; In1.Cu" in content or "F.Cu -> In1.Cu" in content

    def test_report_gap_fallback_indicator(self, basic_report_params):
        """Test that gap fallback is indicated when used."""
        params = basic_report_params.copy()
        params['stackup_derived']['gap_fallback_used'] = True

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "True" in content  # gap_fallback_used should appear

    def test_report_with_snapshot_debug(self, basic_report_params):
        """Test report with snapshot debug information."""
        params = basic_report_params.copy()
        params['snapshot_debug'] = {
            'snap_times': [1.0, 5.0, 10.0],
            'total_captured': 3,
        }

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "Snapshot Debug" in content

    def test_report_contains_initialization_timing_debug(self, basic_report_params):
        """Initialization timing metrics should appear in diagnostics output."""
        params = basic_report_params.copy()
        params['snapshot_debug'] = {
            'init_zone_refill_s': 0.12,
            'init_geometry_maps_s': 0.34,
            'init_capacity_build_s': 0.05,
            'init_power_vector_build_s': 0.02,
            'init_stiffness_matrix_s': 0.17,
        }

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "init_zone_refill_s" in content
        assert "init_geometry_maps_s" in content
        assert "init_stiffness_matrix_s" in content

    def test_report_effective_dielectric_section(self, basic_report_params):
        """Test that effective dielectric thickness section is present."""
        params = basic_report_params.copy()
        params['k_norm_info'] = {
            't_fr4_eff_per_plane_mm': [0.765, 0.765],
        }

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "Effective Dielectric Thickness" in content
        assert "0.765" in content

    def test_report_contains_current_path_diagnostics(self, basic_report_params, temp_dir):
        """Current-path diagnostics should render rich electrical metrics."""
        joule_map = os.path.join(temp_dir, "joule_loss_map.png")
        with open(joule_map, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
        params = basic_report_params.copy()
        params['k_norm_info'] = {
            'grid_requested_res_mm': 0.1,
            'grid_res_mm': 0.184475,
            'grid_auto_coarsened': True,
            'grid_rows': 259,
            'grid_cols': 394,
        }
        params['electrical_summary'] = {
            'total_loss_w': 0.233886,
            'warnings': [],
            'nets': [
                {
                    'net': 'PWR<main>',
                    'net_name': 'PWR<main>',
                    'terminal_count': 2,
                    'source_current_a': 5.0,
                    'sink_current_a': 5.0,
                    'current_balance_a': 0.0,
                    'total_abs_current_a': 10.0,
                    'total_loss_w': 0.233886,
                    'max_node_power_w': 0.000524,
                    'effective_resistance_ohm': 0.009355,
                    'equivalent_voltage_drop_v': 0.046777,
                    'pad_resistance_ohm': 0.009350,
                    'copper_cell_count': 450,
                    'edge_count': 820,
                    'via_edge_count': 0,
                    'connected_component_count': 3,
                    'terminal_diagnostics': [
                        {
                            'name': 'U1<1>',
                            'net_name': 'PWR<main>',
                            'current_a': 5.0,
                            'layer': 'F.Cu',
                            'x_mm': 10.0,
                            'y_mm': 20.0,
                            'bbox_mm': [9.5, 19.5, 1.0, 1.0],
                            'cell_count': 12,
                            'component_ids': [0],
                            'mean_potential_v': 0.046,
                        }
                    ],
                    'primitive_diagnostics': [
                        {
                            'net_name': 'PWR<main>',
                            'primitive_type': 'Track',
                            'layer': 'F.Cu',
                            'count': 1,
                            'track_length_mm': 20.0,
                            'track_width_min_mm': 2.079,
                            'track_width_avg_mm': 2.079,
                            'track_width_max_mm': 2.079,
                            'bbox_area_mm2': 41.584,
                            'mapped_cell_count': 450,
                        }
                    ],
                }
            ],
        }
        params['joule_map_path'] = joule_map

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "Current Path Diagnostics" in content
        assert "Path Current" in content
        assert "Abs Current" not in content
        assert "5.0000 A" in content
        assert "R_eff" in content
        assert "V_eq" in content
        assert "0.184 mm" in content
        assert "requested: 0.100 mm" in content
        assert "Grid resolution was auto-coarsened" in content
        assert "Mapped KiCad Primitives" in content
        assert "joule_loss_map.png" in content
        assert "PWR&lt;main&gt;" in content
        assert "U1&lt;1&gt;" in content


class TestWriteHtmlReportEdgeCases:
    """Edge case tests for write_html_report."""

    def test_empty_pad_power_list(self, temp_dir):
        """Test report with empty pad power list."""
        params = {
            'settings': {},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': None,
                'copper_thickness_mm_used': [],
                'gap_mm_used': [],
                'gap_fallback_used': False,
            },
            'pad_power': [],
            'layer_names': [],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
        }

        result = write_html_report(**params)
        assert result is not None
        assert os.path.exists(result)

    def test_missing_stackup_values(self, temp_dir):
        """Test report handles missing stackup values gracefully."""
        params = {
            'settings': {},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': None,
                'stack_board_thick_mm': None,
                'copper_thickness_mm_used': [],
                'gap_mm_used': [],
                'gap_fallback_used': False,
            },
            'pad_power': [],
            'layer_names': [],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
        }

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "n/a" in content

    def test_special_characters_in_pad_names(self, temp_dir):
        """Test report with special characters in pad names."""
        params = {
            'settings': {},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [],
                'gap_mm_used': [],
                'gap_fallback_used': False,
            },
            'pad_power': [
                ('U1:EP<special>', 1.0),
                ('R&D:1', 0.5),
            ],
            'layer_names': [],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
        }

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        # Special chars should be escaped
        assert '&lt;special&gt;' in content
        assert 'R&amp;D' in content
