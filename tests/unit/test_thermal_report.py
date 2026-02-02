"""
Unit tests for thermal_report module.

This module tests the HTML report generation functionality.
"""

import os
import pytest
import tempfile

from ThermalSim.thermal_report import (
    _fmt,
    _esc,
    write_html_report,
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

        assert "KiCad Thermal Sim Report" in content

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
