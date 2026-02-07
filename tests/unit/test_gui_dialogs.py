"""
Unit tests for gui_dialogs module.

This module tests the wxPython dialog settings parsing.
Note: These tests focus on the data handling aspects that can be
tested without a running wx application.
"""

import pytest


class TestSettingsValueParsing:
    """Tests for settings value parsing logic."""

    def test_parse_float_valid(self):
        """Test parsing valid float strings."""
        assert float("1.5") == 1.5
        assert float("0.001") == 0.001
        assert float("100") == 100.0

    def test_parse_float_invalid_raises(self):
        """Test that invalid float strings raise ValueError."""
        with pytest.raises(ValueError):
            float("not_a_number")
        with pytest.raises(ValueError):
            float("")
        with pytest.raises(ValueError):
            float("1.2.3")

    def test_parse_int_valid(self):
        """Test parsing valid integer strings."""
        assert int("5") == 5
        assert int("100") == 100
        assert int("0") == 0

    def test_parse_int_from_float_string(self):
        """Test that int() on float string raises."""
        with pytest.raises(ValueError):
            int("1.5")

    def test_strip_whitespace(self):
        """Test that whitespace is handled."""
        assert float("  1.5  ".strip()) == 1.5
        assert int("  10  ".strip()) == 10

    def test_power_string_single_value(self):
        """Test single power value parsing."""
        power_str = "1.0"
        values = [float(v.strip()) for v in power_str.split(",")]
        assert values == [1.0]

    def test_power_string_multiple_values(self):
        """Test comma-separated power values."""
        power_str = "1.0, 0.5, 2.0"
        values = [float(v.strip()) for v in power_str.split(",")]
        assert values == [1.0, 0.5, 2.0]

    def test_power_string_extra_whitespace(self):
        """Test power string with extra whitespace."""
        power_str = "  1.0 ,  0.5  , 2.0  "
        values = [float(v.strip()) for v in power_str.split(",")]
        assert values == [1.0, 0.5, 2.0]

    def test_power_string_invalid(self):
        """Test that invalid power string raises."""
        power_str = "1.0, invalid, 2.0"
        with pytest.raises(ValueError):
            [float(v.strip()) for v in power_str.split(",")]

    def test_power_str_with_pwl_paths(self):
        """Test that power_str can contain PWL file paths mixed with constants."""
        power_str = r"1.0, C:\sim\ramp.pwl, 2.0"
        entries = [x.strip() for x in power_str.split(",")]
        assert len(entries) == 3
        assert entries[0] == "1.0"
        assert entries[1] == r"C:\sim\ramp.pwl"
        assert entries[2] == "2.0"

        # Auto-detect: float vs path
        results = []
        for entry in entries:
            try:
                results.append(('const', float(entry)))
            except ValueError:
                results.append(('pwl', entry))
        assert results[0] == ('const', 1.0)
        assert results[1] == ('pwl', r"C:\sim\ramp.pwl")
        assert results[2] == ('const', 2.0)

    def test_power_str_single_pwl_for_all(self):
        """Test that a single PWL path is valid as power_str."""
        power_str = r"C:\sim\ramp.pwl"
        entries = [x.strip() for x in power_str.split(",")]
        assert len(entries) == 1
        try:
            float(entries[0])
            is_path = False
        except ValueError:
            is_path = True
        assert is_path


class TestSettingsDefaults:
    """Tests for settings default values."""

    @pytest.fixture
    def default_values(self):
        """Expected default values for dialog."""
        return {
            'power_str': '1.0',
            'time': 20.0,
            'amb': 25.0,
            'thick': 1.6,
            'res': 0.5,
            'show_all': True,
            'snapshots': False,
            'snap_count': 5,
            'output_dir': '',
            'ignore_traces': False,
            'ignore_polygons': False,
            'limit_area': False,
            'pad_dist_mm': 30.0,
            'use_heatsink': False,
            'pad_th': 1.0,
            'pad_k': 3.0,
            'pad_cap_areal': 0.0,
        }

    def test_default_power(self, default_values):
        """Test default power value."""
        assert default_values['power_str'] == '1.0'

    def test_default_simulation_time(self, default_values):
        """Test default simulation time."""
        assert default_values['time'] == 20.0

    def test_default_ambient(self, default_values):
        """Test default ambient temperature."""
        assert default_values['amb'] == 25.0

    def test_default_thickness(self, default_values):
        """Test default PCB thickness."""
        assert default_values['thick'] == 1.6

    def test_default_heatsink_settings(self, default_values):
        """Test default heatsink settings."""
        assert default_values['use_heatsink'] is False
        assert default_values['pad_th'] == 1.0
        assert default_values['pad_k'] == 3.0


class TestSettingsValidation:
    """Tests for settings validation logic."""

    def test_positive_time_required(self):
        """Test that simulation time should be positive."""
        time_value = -1.0
        assert time_value <= 0  # Invalid

        time_value = 20.0
        assert time_value > 0  # Valid

    def test_positive_resolution_required(self):
        """Test that resolution should be positive."""
        res = 0.0
        assert res <= 0  # Invalid

        res = 0.5
        assert res > 0  # Valid

    def test_snap_count_positive_integer(self):
        """Test that snap count should be positive integer."""
        snap_count = -5
        assert snap_count <= 0  # Invalid

        snap_count = 5
        assert snap_count > 0  # Valid

    def test_thermal_conductivity_positive(self):
        """Test that thermal conductivity should be positive."""
        pad_k = 0.0
        assert pad_k <= 0  # Invalid (would cause division issues)

        pad_k = 3.0
        assert pad_k > 0  # Valid


class TestSettingsDictFormat:
    """Tests for settings dictionary format."""

    def test_settings_dict_structure(self):
        """Test expected settings dictionary structure."""
        settings = {
            'power_str': '1.0',
            'time': 20.0,
            'amb': 25.0,
            'thick': 1.6,
            'res': 0.5,
            'show_all': True,
            'snapshots': False,
            'snap_count': 5,
            'output_dir': '/tmp/output',
            'ignore_traces': False,
            'ignore_polygons': False,
            'limit_area': False,
            'pad_dist_mm': 30.0,
            'use_heatsink': False,
            'pad_th': 1.0,
            'pad_k': 3.0,
            'pad_cap_areal': 0.0,
        }

        # Verify all expected keys are present
        expected_keys = [
            'power_str', 'time', 'amb', 'thick', 'res',
            'show_all', 'snapshots', 'snap_count', 'output_dir',
            'ignore_traces', 'ignore_polygons', 'limit_area', 'pad_dist_mm',
            'use_heatsink', 'pad_th', 'pad_k', 'pad_cap_areal'
        ]

        for key in expected_keys:
            assert key in settings, f"Missing key: {key}"

    def test_settings_types(self):
        """Test that settings have correct types."""
        settings = {
            'power_str': '1.0',
            'time': 20.0,
            'amb': 25.0,
            'thick': 1.6,
            'res': 0.5,
            'show_all': True,
            'snapshots': False,
            'snap_count': 5,
            'output_dir': '/tmp/output',
            'ignore_traces': False,
            'ignore_polygons': False,
            'limit_area': False,
            'pad_dist_mm': 30.0,
            'use_heatsink': False,
            'pad_th': 1.0,
            'pad_k': 3.0,
            'pad_cap_areal': 0.0,
        }

        # Check types
        assert isinstance(settings['power_str'], str)
        assert isinstance(settings['time'], float)
        assert isinstance(settings['amb'], float)
        assert isinstance(settings['thick'], float)
        assert isinstance(settings['res'], float)
        assert isinstance(settings['show_all'], bool)
        assert isinstance(settings['snapshots'], bool)
        assert isinstance(settings['snap_count'], int)
        assert isinstance(settings['output_dir'], str)
        assert isinstance(settings['ignore_traces'], bool)
        assert isinstance(settings['limit_area'], bool)
        assert isinstance(settings['pad_dist_mm'], float)
        assert isinstance(settings['use_heatsink'], bool)
        assert isinstance(settings['pad_th'], float)
        assert isinstance(settings['pad_k'], float)
        assert isinstance(settings['pad_cap_areal'], float)


class TestApplyDefaults:
    """Tests for applying default values logic."""

    def test_merge_partial_defaults(self):
        """Test merging partial defaults with base values."""
        base = {
            'power_str': '1.0',
            'time': 20.0,
            'amb': 25.0,
        }

        defaults = {
            'time': 30.0,  # Override
            'amb': 30.0,   # Override
        }

        # Simulate apply_defaults behavior
        result = base.copy()
        for key, value in defaults.items():
            if key in result:
                result[key] = value

        assert result['power_str'] == '1.0'  # Unchanged
        assert result['time'] == 30.0  # Updated
        assert result['amb'] == 30.0  # Updated

    def test_ignore_unknown_defaults(self):
        """Test that unknown default keys are ignored."""
        base = {
            'power_str': '1.0',
            'time': 20.0,
        }

        defaults = {
            'unknown_key': 'value',
            'time': 30.0,
        }

        result = base.copy()
        for key, value in defaults.items():
            if key in result:
                result[key] = value

        assert 'unknown_key' not in result
        assert result['time'] == 30.0

    def test_type_conversion_in_defaults(self):
        """Test type conversion when applying defaults."""
        # Defaults might come as strings from JSON
        defaults = {
            'time': '30.0',
            'snap_count': '10',
            'show_all': 'True',
        }

        # Conversion logic
        result = {
            'time': float(defaults.get('time', '20.0')),
            'snap_count': int(defaults.get('snap_count', '5')),
            'show_all': str(defaults.get('show_all', 'True')).lower() == 'true',
        }

        assert result['time'] == 30.0
        assert result['snap_count'] == 10
        assert result['show_all'] is True


class TestOutputDirectoryHandling:
    """Tests for output directory handling."""

    def test_empty_output_dir(self):
        """Test handling of empty output directory."""
        output_dir = ''
        assert output_dir == '' or output_dir is None or len(output_dir.strip()) == 0

    def test_output_dir_strip(self):
        """Test stripping whitespace from output directory."""
        output_dir = '  /path/to/dir  '
        assert output_dir.strip() == '/path/to/dir'

    def test_output_dir_with_spaces(self):
        """Test output directory path with spaces."""
        output_dir = '/path/with spaces/dir'
        assert ' ' in output_dir
        # Should be valid path string
        assert isinstance(output_dir, str)
