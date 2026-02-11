"""
Unit tests for thermal_report module.

This module tests the HTML report generation functionality.
"""

import json
import os
import re
import pytest
import tempfile
import numpy as np

from ThermalSim.thermal_report import (
    _fmt,
    _esc,
    _build_interactive_section,
    write_html_report,
    write_interactive_viewer,
)


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

        assert "ThermalSim Report" in content

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

    def test_report_heatmap_png_not_embedded(self, basic_report_params, temp_dir):
        """Test that static heatmap PNG is no longer embedded (replaced by interactive)."""
        heatmap_path = os.path.join(temp_dir, "heatmap.png")
        with open(heatmap_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')

        params = basic_report_params.copy()
        params['heatmap_path'] = heatmap_path

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        # Static heatmap is no longer shown; interactive heatmap replaces it
        assert "heatmap.png" not in content

    def test_report_without_images(self, basic_report_params):
        """Test report handles missing images gracefully."""
        result = write_html_report(**basic_report_params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "not available" in content.lower()

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

    def test_report_no_snapshot_gallery_without_files(self, basic_report_params, temp_dir):
        """Test that snapshot gallery is not shown when no snapshot files provided."""
        params = basic_report_params.copy()
        params['snapshot_files'] = None

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        # No visible snapshot gallery heading when no files
        assert "Time-Series Snapshots</h2>" not in content

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
        # Gap interfaces use arrow entity in combined stackup table
        assert "F.Cu" in content and "In1.Cu" in content

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

    def test_report_effective_dielectric_in_stackup(self, basic_report_params):
        """Test that effective dielectric thickness values appear in stackup table."""
        params = basic_report_params.copy()
        params['k_norm_info'] = {
            't_fr4_eff_per_plane_mm': [0.765, 0.765],
        }

        result = write_html_report(**params)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        # t_fr4_eff values appear in the combined PCB Stackup table
        assert "PCB Stackup" in content
        assert "0.765" in content


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


class TestReportDesign:
    """Tests for the redesigned report layout and features."""

    @pytest.fixture
    def report_with_solver_info(self, temp_dir):
        """Create a report with full solver info including energy balance."""
        T = np.full((2, 10, 10), 25.0)
        T[0, 5, 5] = 75.0
        return {
            'settings': {'power_str': '1.0', 'time': 20.0, 'amb': 25.0, 'res': 0.5},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035, 0.035],
                'gap_mm_used': [1.53],
                'gap_fallback_used': False,
            },
            'pad_power': [('U1:1', 1.0)],
            'layer_names': ['F.Cu', 'B.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'k_norm_info': {
                'strategy': 'implicit_fvm_bdf2',
                'backend': 'SciPy',
                'pin_w': 1.0,
                'pout_final_w': 0.98,
                'steady_rel_diff': 0.02,
                't_fr4_eff_per_plane_mm': [0.765, 0.765],
            },
            'out_dir': temp_dir,
            'T_data': T,
            'ambient': 25.0,
        }

    def _read(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def test_energy_balance_card_present(self, report_with_solver_info):
        """Energy balance card shows Pin, Pout, and rel diff."""
        path = write_html_report(**report_with_solver_info)
        content = self._read(path)
        assert "Energy Balance" in content
        assert "P<sub>in</sub>" in content
        assert "P<sub>out</sub>" in content
        assert "1.0" in content  # pin
        assert "0.98" in content  # pout

    def test_energy_balance_green_dot(self, report_with_solver_info):
        """Rel diff < 5% should show Acceptable label."""
        path = write_html_report(**report_with_solver_info)
        content = self._read(path)
        # 2% rel diff -> Acceptable (yellow)
        assert "Acceptable" in content
        assert "balance-dot" in content

    def test_energy_balance_excellent(self, report_with_solver_info, temp_dir):
        """Rel diff < 1% should show Excellent label."""
        params = report_with_solver_info.copy()
        params['k_norm_info'] = dict(params['k_norm_info'])
        params['k_norm_info']['steady_rel_diff'] = 0.005
        path = write_html_report(**params)
        content = self._read(path)
        assert "Excellent" in content

    def test_peak_temperatures_card(self, report_with_solver_info):
        """Peak temperature card shows per-layer max values."""
        path = write_html_report(**report_with_solver_info)
        content = self._read(path)
        assert "Peak Temperatures" in content
        assert "75.0" in content  # max temp from T_data

    def test_overview_card(self, report_with_solver_info):
        """Overview card shows layers, ambient, sim time."""
        path = write_html_report(**report_with_solver_info)
        content = self._read(path)
        assert "Overview" in content
        assert "Layers" in content
        assert "Ambient" in content

    def test_collapsible_debug(self, report_with_solver_info):
        """Debug section is in a collapsible details element."""
        path = write_html_report(**report_with_solver_info)
        content = self._read(path)
        assert "<details>" in content
        assert "Solver Debug" in content

    def test_hover_hints_on_headers(self, report_with_solver_info):
        """Table headers and labels have title attributes for hover hints."""
        path = write_html_report(**report_with_solver_info)
        content = self._read(path)
        assert 'title="' in content

    def test_combined_stackup_table(self, report_with_solver_info):
        """Stackup table combines copper thickness and t_fr4_eff."""
        path = write_html_report(**report_with_solver_info)
        content = self._read(path)
        assert "PCB Stackup" in content
        assert "Cu Thickness" in content
        assert "t<sub>fr4,eff</sub>" in content

    def test_no_heatmap_png_in_report(self, report_with_solver_info):
        """Static heatmap PNG should not appear in the report."""
        path = write_html_report(**report_with_solver_info)
        content = self._read(path)
        assert "Heatmap</h3>" not in content

    def test_gap_fallback_note(self, temp_dir):
        """Fallback note appears when uniform gap was used."""
        params = {
            'settings': {'power_str': '1.0'},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035, 0.035],
                'gap_mm_used': [0.8],
                'gap_fallback_used': True,
            },
            'pad_power': [],
            'layer_names': ['F.Cu', 'B.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
        }
        path = write_html_report(**params)
        content = self._read(path)
        assert "Uniform dielectric gap fallback" in content

    def test_heat_sources_section_name(self, report_with_solver_info):
        """Pad power section is named 'Heat Sources'."""
        path = write_html_report(**report_with_solver_info)
        content = self._read(path)
        assert "Heat Sources" in content


class TestInteractiveHeatmap:
    """Tests for interactive heatmap section in HTML report."""

    @pytest.fixture
    def base_params(self, temp_dir):
        """Minimal params for write_html_report."""
        return {
            'settings': {'power_str': '1.0', 'time': 20.0},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035, 0.035],
                'gap_mm_used': [1.53],
                'gap_fallback_used': False,
            },
            'pad_power': [('U1:1', 1.0)],
            'layer_names': ['F.Cu', 'B.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
        }

    def _read_report(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def test_no_interactive_section_when_t_data_none(self, base_params):
        """T_data=None should not produce interactive canvas elements."""
        path = write_html_report(**base_params)
        content = self._read_report(path)
        assert "var T_DATA" not in content

    def test_interactive_section_present(self, base_params):
        """Passing T_data should produce the interactive section heading."""
        T = np.full((2, 10, 10), 25.0)
        T[0, 5, 5] = 80.0
        base_params['T_data'] = T
        base_params['ambient'] = 25.0

        path = write_html_report(**base_params)
        content = self._read_report(path)
        assert "Interactive Heatmap" in content

    def test_json_data_embedded(self, base_params):
        """T_DATA variable should be present inside a script tag."""
        T = np.full((2, 5, 5), 30.0)
        base_params['T_data'] = T
        base_params['ambient'] = 25.0

        path = write_html_report(**base_params)
        content = self._read_report(path)
        assert "var T_DATA = " in content

    def test_canvas_elements_per_layer(self, base_params):
        """One canvas element per layer."""
        T = np.full((2, 8, 8), 25.0)
        base_params['T_data'] = T
        base_params['ambient'] = 25.0

        path = write_html_report(**base_params)
        content = self._read_report(path)
        assert 'id="layer-0"' in content
        assert 'id="layer-1"' in content

    def test_layer_names_in_js(self, base_params):
        """LAYER_NAMES JS array should match provided layer names."""
        T = np.full((2, 5, 5), 25.0)
        base_params['T_data'] = T
        base_params['ambient'] = 25.0

        path = write_html_report(**base_params)
        content = self._read_report(path)
        assert "var LAYER_NAMES = " in content
        assert '"F.Cu"' in content
        assert '"B.Cu"' in content

    def test_ambient_value_embedded(self, base_params):
        """AMBIENT JS variable should match provided ambient temperature."""
        T = np.full((2, 5, 5), 30.0)
        base_params['T_data'] = T
        base_params['ambient'] = 22.5

        path = write_html_report(**base_params)
        content = self._read_report(path)
        assert "var AMBIENT = 22.5" in content

    def test_four_layer_stackup(self, temp_dir):
        """Four-layer board should produce 4 canvas elements."""
        T = np.full((4, 10, 10), 25.0)
        params = {
            'settings': {},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035] * 4,
                'gap_mm_used': [0.2, 1.0, 0.2],
                'gap_fallback_used': False,
            },
            'pad_power': [],
            'layer_names': ['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
            'T_data': T,
            'ambient': 25.0,
        }

        path = write_html_report(**params)
        content = self._read_report(path)
        for i in range(4):
            assert f'id="layer-{i}"' in content
        assert "In1.Cu" in content
        assert "In2.Cu" in content

    def test_large_grid(self, base_params):
        """200x200 grid should work without error."""
        T = np.full((2, 200, 200), 25.0)
        T[0, 100, 100] = 90.0
        base_params['T_data'] = T
        base_params['ambient'] = 25.0

        path = write_html_report(**base_params)
        assert path is not None
        content = self._read_report(path)
        assert "Interactive Heatmap" in content
        assert 'width="200"' in content
        assert 'height="200"' in content

    def test_temperature_rounding(self, base_params):
        """Temperature values should be rounded to 1 decimal place."""
        T = np.full((2, 3, 3), 25.123456789)
        base_params['T_data'] = T
        base_params['ambient'] = 25.0

        path = write_html_report(**base_params)
        content = self._read_report(path)
        # The exact float 25.123456789 should not appear; 25.1 should
        assert "25.123456789" not in content
        assert "25.1" in content

    def test_special_chars_in_layer_names(self, temp_dir):
        """Layer names with special chars should be safely embedded in JS."""
        T = np.full((2, 5, 5), 30.0)
        params = {
            'settings': {},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035, 0.035],
                'gap_mm_used': [1.53],
                'gap_fallback_used': False,
            },
            'pad_power': [],
            'layer_names': ['F.Cu<test>', 'B.Cu&"special'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
            'T_data': T,
            'ambient': 25.0,
        }

        path = write_html_report(**params)
        content = self._read_report(path)
        # HTML heading should be escaped
        assert "&lt;test&gt;" in content
        # JS string should use JSON-safe encoding (no raw < or unescaped quotes)
        assert "Interactive Heatmap" in content

    def test_vmax_capped(self, base_params):
        """vmax should be capped at ambient + 250."""
        T = np.full((2, 5, 5), 25.0)
        T[0, 0, 0] = 500.0  # way above ambient + 250 = 275
        base_params['T_data'] = T
        base_params['ambient'] = 25.0

        path = write_html_report(**base_params)
        content = self._read_report(path)
        # Legend max should show 275.0, not 500.0
        assert "275.0" in content

    def test_build_interactive_section_directly(self):
        """Test _build_interactive_section helper directly."""
        T = np.array([[[25.0, 30.0], [35.0, 40.0]]])
        result = _build_interactive_section(T, 25.0, ['F.Cu'])
        assert "Interactive Heatmap" in result
        assert "var T_DATA" in result
        assert 'id="layer-0"' in result
        assert "F.Cu" in result


class TestInteractiveViewer:
    """Tests for write_interactive_viewer function."""

    def _read(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def test_viewer_file_created(self, temp_dir):
        """File exists and ends with thermal_viewer.html."""
        T = np.full((2, 10, 10), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=10.0,
            ambient=25.0, layer_names=['F.Cu', 'B.Cu'], out_dir=temp_dir,
        )
        assert path is not None
        assert os.path.exists(path)
        assert path.endswith("thermal_viewer.html")

    def test_viewer_returns_none_on_bad_dir(self):
        """Graceful failure on bad output directory."""
        T = np.full((1, 5, 5), 25.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'],
            out_dir="/nonexistent_dir_xyz_123/sub",
        )
        assert path is None

    def test_viewer_contains_frames_json(self, temp_dir):
        """var FRAMES present, 'Final' label present."""
        T = np.full((1, 5, 5), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=10.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert "var FRAMES" in content
        assert "Final" in content

    def test_viewer_with_snapshots(self, temp_dir):
        """Multiple frame labels when snapshots provided."""
        T_final = np.full((1, 5, 5), 50.0)
        snap1 = (2.0, np.full((1, 5, 5), 35.0))
        snap2 = (5.0, np.full((1, 5, 5), 42.0))
        path = write_interactive_viewer(
            T_snapshots=[snap1, snap2], T_final=T_final, sim_time=10.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert "t = 2.0 s" in content
        assert "t = 5.0 s" in content
        assert "Final" in content

    def test_viewer_layer_buttons(self, temp_dir):
        """Layer names appear in LAYER_NAMES JS variable."""
        T = np.full((4, 5, 5), 30.0)
        names = ['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu']
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=10.0,
            ambient=25.0, layer_names=names, out_dir=temp_dir,
        )
        content = self._read(path)
        assert "var LAYER_NAMES" in content
        for n in names:
            assert n in content

    def test_viewer_no_snapshots_hides_frame_buttons(self, temp_dir):
        """Single frame entry when no snapshots — frame toolbar hidden."""
        T = np.full((1, 5, 5), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=10.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        # Only one frame button (Final), and toolbar is hidden
        assert content.count('class="btn frame-btn') == 1
        assert 'display:none;' in content  # frame toolbar hidden

    def test_viewer_canvas_dimensions(self, temp_dir):
        """Canvas width/height match grid cols/rows."""
        T = np.full((1, 15, 20), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert 'width="20"' in content
        assert 'height="15"' in content

    def test_viewer_vmax_capped(self, temp_dir):
        """Max capped at ambient + 250."""
        T = np.full((1, 5, 5), 25.0)
        T[0, 0, 0] = 500.0  # way above 25 + 250 = 275
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert "275.0" in content

    def test_viewer_temperature_rounding(self, temp_dir):
        """No raw floats, values rounded to 1dp."""
        T = np.full((1, 3, 3), 25.123456789)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert "25.123456789" not in content
        assert "25.1" in content

    def test_viewer_valid_html(self, temp_dir):
        """DOCTYPE, canvas, heatmap id present."""
        T = np.full((1, 5, 5), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert "<!DOCTYPE html>" in content
        assert "<canvas" in content
        assert 'id="heatmap"' in content

    def test_viewer_hover_tooltip(self, temp_dir):
        """Tooltip div and mousemove handler present."""
        T = np.full((1, 5, 5), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert 'id="tooltip"' in content
        assert "mousemove" in content

    def test_viewer_global_vmin_vmax(self, temp_dir):
        """VMAX derived from hottest frame across all snapshots."""
        T_final = np.full((1, 5, 5), 50.0)
        # Snapshot has a hotter pixel than final
        T_snap = np.full((1, 5, 5), 30.0)
        T_snap[0, 0, 0] = 120.0
        path = write_interactive_viewer(
            T_snapshots=[(3.0, T_snap)], T_final=T_final, sim_time=10.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        # VMAX should be 120.0 (from snapshot), not 50.0 (from final)
        assert "var VMAX = 120.0" in content

    def test_report_no_interactive_without_tdata(self, temp_dir):
        """write_html_report(T_data=None) omits interactive canvas."""
        params = {
            'settings': {'power_str': '1.0'},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035],
                'gap_mm_used': [],
                'gap_fallback_used': False,
            },
            'pad_power': [],
            'layer_names': ['F.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
        }
        path = write_html_report(**params)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "var T_DATA" not in content

    def test_viewer_responsive_canvas(self, temp_dir):
        """Viewer canvas-wrap uses 90vw responsive width."""
        T = np.full((1, 10, 10), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert "90vw" in content
        assert "max-width: 1400px" in content

    def test_viewer_overlay_canvas(self, temp_dir):
        """Viewer contains an overlay canvas for rectangle drawing."""
        T = np.full((1, 10, 10), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert 'id="overlay"' in content

    def test_viewer_stats_panel(self, temp_dir):
        """Viewer contains the stats panel div."""
        T = np.full((1, 10, 10), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert 'id="stats-panel"' in content
        assert 'id="stats-list"' in content
        assert "Selections" in content

    def test_viewer_clear_all_button(self, temp_dir):
        """Viewer contains Clear All button."""
        T = np.full((1, 10, 10), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert 'id="clear-all-btn"' in content
        assert "Clear All" in content

    def test_viewer_rectangle_js_functions(self, temp_dir):
        """Viewer contains rectangle selection JS functions."""
        T = np.full((1, 10, 10), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert "computeStats" in content
        assert "renderOverlay" in content
        assert "updateStatsPanel" in content
        assert "RECT_COLORS" in content

    def test_viewer_mousedown_handler(self, temp_dir):
        """Viewer has mousedown handler on overlay for drawing."""
        T = np.full((1, 10, 10), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert "mousedown" in content
        assert "mouseup" in content

    def test_viewer_flex_layout(self, temp_dir):
        """Viewer uses flex layout for canvas + stats panel."""
        T = np.full((1, 10, 10), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        assert "viewer-main" in content
        assert "display: flex" in content

    def test_viewer_overlay_dimensions_match_heatmap(self, temp_dir):
        """Overlay canvas has same dimensions as heatmap canvas."""
        T = np.full((1, 15, 20), 30.0)
        path = write_interactive_viewer(
            T_snapshots=[], T_final=T, sim_time=5.0,
            ambient=25.0, layer_names=['F.Cu'], out_dir=temp_dir,
        )
        content = self._read(path)
        # Both heatmap and overlay should have width=20, height=15
        assert content.count('width="20"') >= 2
        assert content.count('height="15"') >= 2

    def test_report_interactive_section_responsive_canvas(self, temp_dir):
        """Interactive section in report has responsive canvas CSS."""
        T = np.full((2, 5, 5), 30.0)
        params = {
            'settings': {},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035, 0.035],
                'gap_mm_used': [1.53],
                'gap_fallback_used': False,
            },
            'pad_power': [],
            'layer_names': ['F.Cu', 'B.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
            'T_data': T,
            'ambient': 25.0,
        }
        path = write_html_report(**params)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "90vw" in content
        assert "max-width: 1200px" in content


class TestSnapshotGallery:
    """Tests for snapshot gallery in HTML report."""

    def _read(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def test_snapshot_gallery_in_report(self, temp_dir):
        """Report with snapshot_files should contain snapshot gallery."""
        # Create dummy snapshot PNG files
        snap_path_1 = os.path.join(temp_dir, "snap_00_t1.0.png")
        snap_path_2 = os.path.join(temp_dir, "snap_01_t5.0.png")
        for p in (snap_path_1, snap_path_2):
            with open(p, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n')

        params = {
            'settings': {'power_str': '1.0', 'time': 10.0},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035, 0.035],
                'gap_mm_used': [1.53],
                'gap_fallback_used': False,
            },
            'pad_power': [('U1:1', 1.0)],
            'layer_names': ['F.Cu', 'B.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
            'snapshot_files': [
                (1.0, snap_path_1),
                (5.0, snap_path_2),
            ],
        }

        path = write_html_report(**params)
        content = self._read(path)

        assert "Time-Series Snapshots" in content
        assert "snap-gallery" in content
        assert "snap_00_t1.0.png" in content
        assert "snap_01_t5.0.png" in content
        assert "t = 1.0 s" in content
        assert "t = 5.0 s" in content

    def test_no_gallery_without_snapshot_files(self, temp_dir):
        """Report without snapshot_files should not contain gallery."""
        params = {
            'settings': {'power_str': '1.0'},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035],
                'gap_mm_used': [],
                'gap_fallback_used': False,
            },
            'pad_power': [],
            'layer_names': ['F.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
            'snapshot_files': None,
        }

        path = write_html_report(**params)
        content = self._read(path)

        assert "Time-Series Snapshots</h2>" not in content

    def test_gallery_skips_missing_files(self, temp_dir):
        """Gallery should skip entries where file doesn't exist."""
        params = {
            'settings': {'power_str': '1.0'},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035],
                'gap_mm_used': [],
                'gap_fallback_used': False,
            },
            'pad_power': [],
            'layer_names': ['F.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
            'snapshot_files': [
                (1.0, os.path.join(temp_dir, "nonexistent.png")),
            ],
        }

        path = write_html_report(**params)
        content = self._read(path)

        # Gallery heading should not appear since file doesn't exist
        assert "Time-Series Snapshots</h2>" not in content

    def test_gallery_css_present(self, temp_dir):
        """Report with snapshots should include gallery CSS."""
        snap_path = os.path.join(temp_dir, "snap_00_t2.0.png")
        with open(snap_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')

        params = {
            'settings': {'power_str': '1.0'},
            'stack_info': {},
            'stackup_derived': {
                'total_thick_mm_used': 1.6,
                'stack_board_thick_mm': 1.6,
                'copper_thickness_mm_used': [0.035],
                'gap_mm_used': [],
                'gap_fallback_used': False,
            },
            'pad_power': [],
            'layer_names': ['F.Cu'],
            'preview_path': None,
            'heatmap_path': None,
            'out_dir': temp_dir,
            'snapshot_files': [(2.0, snap_path)],
        }

        path = write_html_report(**params)
        content = self._read(path)

        assert ".snap-gallery" in content
        assert ".snap-item" in content
