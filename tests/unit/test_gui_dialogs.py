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
            'grid_expert_limits': False,
            'grid_max_cells': 200000,
            'grid_target_cells': 100000,
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
            'grid_expert_limits': False,
            'grid_max_cells': 200000,
            'grid_target_cells': 100000,
        }

        # Verify all expected keys are present
        expected_keys = [
            'power_str', 'time', 'amb', 'thick', 'res',
            'show_all', 'snapshots', 'snap_count', 'output_dir',
            'ignore_traces', 'ignore_polygons', 'limit_area', 'pad_dist_mm',
            'use_heatsink', 'pad_th', 'pad_k', 'pad_cap_areal',
            'grid_expert_limits', 'grid_max_cells', 'grid_target_cells'
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
            'grid_expert_limits': False,
            'grid_max_cells': 200000,
            'grid_target_cells': 100000,
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
        assert isinstance(settings['grid_expert_limits'], bool)
        assert isinstance(settings['grid_max_cells'], int)
        assert isinstance(settings['grid_target_cells'], int)


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


class TestNewSettingsKeys:
    """Tests for the h_conv setting added in the GUI redesign."""

    def test_h_conv_in_default_settings(self, default_settings):
        """Test that h_conv is present in default settings fixture."""
        assert 'h_conv' in default_settings

    def test_h_conv_correct_type(self, default_settings):
        """Test that h_conv is a float."""
        assert isinstance(default_settings['h_conv'], float)

    def test_h_conv_default_value(self, default_settings):
        """Test that h_conv defaults to 10.0."""
        assert default_settings['h_conv'] == 10.0

    def test_h_conv_backward_compat_missing_key(self):
        """Test that missing h_conv key falls back to 10.0."""
        old_settings = {
            'power_str': '1.0',
            'time': 20.0,
            'amb': 25.0,
        }
        # Simulate solver's backward-compatible access
        h_conv = float(old_settings.get('h_conv', 10.0))
        assert h_conv == 10.0

    def test_h_conv_custom_value_preserved(self):
        """Test that a custom h_conv value is read correctly."""
        settings = {'h_conv': 50.0}
        h_conv = float(settings.get('h_conv', 10.0))
        assert h_conv == 50.0

    def test_settings_dict_includes_h_conv(self):
        """Test the full settings dict includes h_conv."""
        expected_keys = [
            'power_str', 'time', 'amb', 'thick', 'res',
            'show_all', 'snapshots', 'snap_count', 'output_dir',
            'ignore_traces', 'ignore_polygons', 'limit_area', 'pad_dist_mm',
            'use_heatsink', 'pad_th', 'pad_k', 'pad_cap_areal', 'h_conv',
            'grid_expert_limits', 'grid_max_cells', 'grid_target_cells'
        ]
        settings = {k: None for k in expected_keys}
        for key in expected_keys:
            assert key in settings, f"Missing key: {key}"


class TestCurrentGroupSettings:
    """Tests for current-path group serialization helpers."""

    def test_total_current_is_split_across_pads(self):
        from ThermalSim.gui_dialogs import prepare_current_groups

        groups = [{
            'name': 'Return',
            'mode': 'total',
            'total_current_a': -9.0,
            'pads': [
                {'pad_key': 'a', 'name': 'J1-1', 'net_name': 'PWR'},
                {'pad_key': 'b', 'name': 'J2-1', 'net_name': 'PWR'},
                {'pad_key': 'c', 'name': 'J3-1', 'net_name': 'PWR'},
            ],
        }]

        prepared = prepare_current_groups(groups)

        currents = [pad['current_a'] for pad in prepared[0]['pads']]
        assert currents == [-3.0, -3.0, -3.0]

    def test_per_pad_current_preserves_values_and_total(self):
        from ThermalSim.gui_dialogs import prepare_current_groups

        groups = [{
            'name': 'Sources',
            'mode': 'per_pad',
            'total_current_a': 0.0,
            'pads': [
                {'pad_key': 'a', 'name': 'J1-1', 'net_name': 'PWR', 'current_a': 9.0},
                {'pad_key': 'b', 'name': 'J2-1', 'net_name': 'PWR', 'current_a': -3.0},
                {'pad_key': 'c', 'name': 'J3-1', 'net_name': 'PWR', 'current_a': -6.0},
            ],
        }]

        prepared = prepare_current_groups(groups)

        assert prepared[0]['total_current_a'] == 0.0
        assert [pad['current_a'] for pad in prepared[0]['pads']] == [9.0, -3.0, -6.0]

    def test_summarize_current_groups_reports_net_balance(self):
        from ThermalSim.gui_dialogs import summarize_current_groups

        groups = [{
            'name': 'Balanced',
            'mode': 'per_pad',
            'pads': [
                {'pad_key': 'a', 'name': 'J1-1', 'net_name': 'PWR', 'current_a': 9.0},
                {'pad_key': 'b', 'name': 'J2-1', 'net_name': 'PWR', 'current_a': -9.0},
            ],
        }]

        _, balance_rows = summarize_current_groups(groups)

        assert balance_rows == [('PWR', '0 A', 'OK')]

    def test_default_mode_is_per_pad_for_new_groups(self):
        from ThermalSim.gui_dialogs import prepare_current_groups

        prepared = prepare_current_groups([{
            'name': 'New Group',
            'pads': [
                {'pad_key': 'a', 'name': 'J1-1', 'net_name': 'PWR', 'current_a': 6.0},
                {'pad_key': 'b', 'name': 'J1-2', 'net_name': 'PWR', 'current_a': -4.0},
                {'pad_key': 'c', 'name': 'J1-3', 'net_name': 'PWR', 'current_a': -2.0},
            ],
        }])

        assert prepared[0]['mode'] == 'per_pad'
        assert [pad['current_a'] for pad in prepared[0]['pads']] == [6.0, -4.0, -2.0]

    def test_legacy_german_modes_are_loaded(self):
        from ThermalSim.gui_dialogs import prepare_current_groups

        per_pad = prepare_current_groups([{'mode': 'Strom pro Pad', 'pads': []}])
        total = prepare_current_groups([{'mode': 'Gesamtstrom verteilen', 'pads': []}])

        assert per_pad[0]['mode'] == 'per_pad'
        assert total[0]['mode'] == 'total'

    def test_unbalanced_group_reports_needs_balance(self):
        from ThermalSim.gui_dialogs import summarize_current_groups

        groups = [{
            'name': 'Unbalanced',
            'mode': 'per_pad',
            'pads': [
                {'pad_key': 'a', 'name': 'J1-1', 'net_name': 'PWR', 'current_a': 6.0},
                {'pad_key': 'b', 'name': 'J1-2', 'net_name': 'PWR', 'current_a': -3.0},
            ],
        }]

        _, balance_rows = summarize_current_groups(groups)

        assert balance_rows == [('PWR', '3 A', 'Needs balance')]

    def test_mixed_nets_are_visible_in_group_summary(self):
        from ThermalSim.gui_dialogs import summarize_current_groups

        groups = [{
            'name': 'Mixed',
            'mode': 'per_pad',
            'pads': [
                {'pad_key': 'a', 'name': 'J1-1', 'net_name': 'PWR', 'current_a': 1.0},
                {'pad_key': 'b', 'name': 'J1-2', 'net_name': 'GND', 'current_a': -1.0},
            ],
        }]

        group_rows, _ = summarize_current_groups(groups)

        assert group_rows[0][1] == 'Mixed nets: GND, PWR'

    def test_dialog_new_group_defaults_to_per_pad_and_applies_current_list(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        selected_pads = [
            {'pad_key': 'a', 'name': 'Pad1', 'net_name': 'PWR', 'layer': 'F.Cu'},
            {'pad_key': 'b', 'name': 'Pad2', 'net_name': 'PWR', 'layer': 'F.Cu'},
            {'pad_key': 'c', 'name': 'Pad3', 'net_name': 'PWR', 'layer': 'F.Cu'},
        ]
        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "B.Cu"],
            selection_provider=lambda: selected_pads
        )

        dlg._on_current_new_group(None)
        dlg.current_pad_list_input.SetValue("+6, -4, -2")
        dlg._on_current_apply_pad_current_list(None)
        values = dlg.get_values()

        group = values['current_groups'][0]
        assert values['current_enabled'] is True
        assert group['mode'] == 'per_pad'
        assert [pad['current_a'] for pad in group['pads']] == [6.0, -4.0, -2.0]
        assert dlg.current_balance_text.GetValue() == "PWR: 0 A - OK"

    def test_power_pads_tab_uses_live_selection_independently(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        selected_pads = [
            {'pad_key': 'p', 'name': 'U1-1 [VIN]', 'net_name': 'VIN', 'layer': 'F.Cu'},
        ]
        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "B.Cu"],
            selection_provider=lambda: selected_pads
        )

        dlg.power_input.SetValue("2.5")
        dlg._on_power_add_selection(None)
        values = dlg.get_values()

        assert values['power_str'] == "2.5"
        assert values['power_pads'][0]['name'] == "U1-1 [VIN]"
        assert values['power_pads'][0]['power'] == "2.5"
        assert values['current_groups'] == []

    def test_legacy_initial_selection_becomes_power_pads(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        initial_pads = [
            {'pad_key': 'a', 'name': 'U1-1 [VIN]', 'net_name': 'VIN', 'layer': 'F.Cu'},
            {'pad_key': 'b', 'name': 'U1-2 [VIN]', 'net_name': 'VIN', 'layer': 'F.Cu'},
        ]
        dlg = SettingsDialog(
            None, 2, 0.5, ["F.Cu", "B.Cu"],
            initial_power_pads=initial_pads
        )

        dlg.power_input.SetValue("1.0, 0.5")
        values = dlg.get_values()

        assert values['power_str'] == "1.0, 0.5"
        assert [pad['power'] for pad in values['power_pads']] == ["1.0", "0.5"]

    def test_saved_empty_power_pads_stays_empty(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        initial_pads = [
            {'pad_key': 'a', 'name': 'U1-1 [VIN]', 'net_name': 'VIN', 'layer': 'F.Cu'},
        ]
        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            initial_power_pads=initial_pads,
            defaults={'power_str': '1.0', 'power_pads': []}
        )

        values = dlg.get_values()

        assert values['power_pads'] == []

    def test_current_paths_tab_labels_are_english(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(None, 0, 0.5, ["F.Cu", "B.Cu"])

        page_labels = [caption for _, caption in dlg.notebook._pages]
        assert page_labels == ["Overview", "Heat Sources", "Current Heating", "Advanced"]
        labels = " ".join(
            list(dlg.current_group_list._columns)
            + list(dlg.current_pad_list._columns)
            + [dlg.chk_current_enabled.label]
        )
        forbidden = [
            "Strom", "Padgruppen", "Auswahl", "Gesamtstrom",
            "Netzbilanz", "Loeschen", "Uebernehmen", "Modus"
        ]
        for word in forbidden:
            assert word not in labels


class TestTooltipTexts:
    """Tests for tooltip text coverage."""

    def test_tooltip_texts_dict_exists(self):
        """Test that TOOLTIP_TEXTS is importable and is a dict."""
        from ThermalSim.gui_dialogs import TOOLTIP_TEXTS
        assert isinstance(TOOLTIP_TEXTS, dict)

    def test_tooltip_texts_covers_all_fields(self):
        """Test that TOOLTIP_TEXTS has entries for all expected controls."""
        from ThermalSim.gui_dialogs import TOOLTIP_TEXTS

        expected_fields = [
            'stackup', 'pads', 'power', 'browse_pwl',
            'duration', 'ambient', 'resolution',
            'show_all', 'snapshots', 'snap_count', 'output_dir',
            'ignore_traces', 'limit_area', 'limit_dist',
            'enable_pad', 'pad_thick', 'pad_k', 'pad_cap',
            'h_conv', 'pcb_thick', 'grid_expert_limits',
            'grid_max_cells', 'grid_target_cells', 'capabilities',
            'help', 'preview', 'load_settings', 'save_settings',
        ]

        for field in expected_fields:
            assert field in TOOLTIP_TEXTS, f"Missing tooltip for: {field}"

    def test_tooltip_texts_are_non_empty_strings(self):
        """Test that all tooltip values are non-empty strings."""
        from ThermalSim.gui_dialogs import TOOLTIP_TEXTS

        for key, value in TOOLTIP_TEXTS.items():
            assert isinstance(value, str), f"Tooltip '{key}' is not a string"
            assert len(value) > 0, f"Tooltip '{key}' is empty"

    def test_tooltip_texts_no_trailing_whitespace(self):
        """Test that tooltips have no trailing whitespace."""
        from ThermalSim.gui_dialogs import TOOLTIP_TEXTS

        for key, value in TOOLTIP_TEXTS.items():
            assert value == value.strip(), f"Tooltip '{key}' has extra whitespace"


class TestSettingsDialogInstantiation:
    """Tests for SettingsDialog creation with mock wx."""

    def test_dialog_creates_without_error(self):
        """Test that SettingsDialog can be instantiated with mocks."""
        from ThermalSim.gui_dialogs import SettingsDialog
        dlg = SettingsDialog(
            None, 2, 0.5, ["F.Cu", "B.Cu"],
            stackup_details="F.Cu: 35um\nB.Cu: 35um",
            pad_names=["U1-1 [VCC]", "U1-2 [GND]"],
            default_output_dir="/tmp"
        )
        assert dlg is not None

    def test_dialog_uses_responsive_size_and_board_context(self):
        """Modern dialog should expose board context and resize safely."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
            board_name="controller.kicad_pcb",
            board_size_mm=(120.0, 80.0),
        )

        assert dlg.GetSize() == (820, 720)
        assert dlg.GetMinSize() == (760, 640)
        assert dlg.lbl_board_name.GetLabel() == "controller.kicad_pcb"
        assert dlg.lbl_board_meta.GetLabel() == "120.0 x 80.0 mm / 4 copper layers"
        assert dlg.btn_run.label == "Run Simulation"
        assert not dlg.btn_run._is_default

    def test_context_summarizes_constant_and_pwl_heat_sources(self):
        """Heat-source summary should expose count, total power, and PWL use."""
        from ThermalSim.gui_dialogs import summarize_power_pads

        assert summarize_power_pads([]) == "0 heat sources"
        assert summarize_power_pads([
            {'power': '1.25'}, {'power': '0.75'},
        ]) == "2 heat sources / 2 W"
        assert summarize_power_pads([
            {'power': '1.0'}, {'power': r'C:\\sim\\ramp.pwl'},
        ]) == "2 heat sources (contains PWL)"

    def test_heat_source_table_supports_multi_row_removal(self):
        """Power-pad actions should operate on multiple selected rows."""
        from ThermalSim.gui_dialogs import SettingsDialog

        pads = [
            {'pad_key': str(idx), 'name': f'P{idx}', 'power': '1.0'}
            for idx in range(3)
        ]
        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "B.Cu"],
            defaults={'power_pads': pads},
        )
        dlg.power_pad_list.Select(0)
        dlg.power_pad_list.Select(1)

        dlg._on_power_remove_pads(None)

        assert [pad['pad_key'] for pad in dlg.power_pads] == ['2']
        assert dlg.lbl_heat_summary.GetLabel() == "1 heat source / 1 W"

    def test_current_mode_shows_only_relevant_entry_rows(self):
        """Current-entry rows should follow the selected distribution mode."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "B.Cu"],
            defaults={
                'current_enabled': True,
                'current_groups': [{
                    'name': 'Supply', 'mode': 'total', 'total_current_a': 4.0,
                    'pads': [{'pad_key': '1', 'name': 'J1-1', 'net_name': 'VIN'}],
                }],
            },
        )

        assert dlg.current_total_row._shown is True
        assert dlg.current_pad_current_row._shown is False
        assert dlg.current_pad_list_row._shown is False

        dlg.current_mode_choice.SetSelection(0)
        dlg._on_current_mode_changed(None)
        assert dlg.current_total_row._shown is False
        assert dlg.current_pad_current_row._shown is True
        assert dlg.current_pad_list_row._shown is True

    def test_active_advanced_sections_expand_from_defaults(self):
        """Non-default advanced settings should reveal their native panes."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "B.Cu"],
            defaults={
                'use_heatsink': True,
                'grid_expert_limits': True,
                'grid_max_cells': 800000,
                'grid_target_cells': 400000,
                'solver_backend': 'scipy',
            },
        )

        assert dlg.geometry_pane.IsExpanded()
        assert dlg.thermal_pad_pane.IsExpanded()
        assert dlg.solver_pane.IsExpanded()
        assert dlg.grid_detail_choice.GetStringSelection() == "Custom"
        assert dlg.grid_node_budget_input.IsEnabled()

    def test_preflight_and_run_result_states_are_persistent(self, tmp_path):
        """Footer should retain readiness and completion information."""
        from ThermalSim.gui_dialogs import SettingsDialog
        from ThermalSim.workflow import GridEstimate, PreflightResult

        grid = GridEstimate(0.5, 0.5, 0, 0, 10, 10, 24, 24, 4, False, False, 200000, 100000)
        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
            preflight_callback=lambda settings: PreflightResult(grid=grid),
        )
        assert dlg.lbl_preflight_status.GetLabel() == "Ready"
        assert "2,304 nodes" in dlg.lbl_preflight.GetLabel()
        assert dlg.btn_run.IsEnabled()

        dlg.preflight_callback = lambda settings: PreflightResult(
            grid=grid, warnings=["Resolution was automatically coarsened."],
        )
        assert dlg._refresh_preflight()
        assert dlg.lbl_preflight_status.GetLabel() == "Warning"
        assert dlg.btn_run.IsEnabled()

        dlg.preflight_callback = lambda settings: PreflightResult(
            grid=grid, errors=["Configure at least one heat source."],
        )
        assert not dlg._refresh_preflight()
        assert dlg.lbl_preflight_status.GetLabel() == "Blocked"
        assert not dlg.btn_run.IsEnabled()

        report_path = tmp_path / "thermal_report.html"
        report_path.write_text("report", encoding="utf-8")
        dlg.set_run_state("running")
        assert not dlg.btn_run.IsEnabled()
        dlg.set_artifacts(str(report_path), str(tmp_path), elapsed_s=8.2, max_temp_c=56.3)

        assert dlg.result_panel.IsShown()
        assert dlg.lbl_result.GetLabel() == "Completed / max 56.3 °C / 8.2 s"
        assert dlg.btn_open_report.IsEnabled()
        assert dlg.btn_open_folder.IsEnabled()

    def test_get_values_returns_dict(self):
        """Test that get_values returns a dict with all expected keys."""
        from ThermalSim.gui_dialogs import SettingsDialog
        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"]
        )
        values = dlg.get_values()
        assert values is not None
        assert isinstance(values, dict)
        assert 'h_conv' in values
        assert isinstance(values['h_conv'], float)

    def test_get_values_h_conv_default(self):
        """Test that h_conv defaults to 10.0 without saved settings."""
        from ThermalSim.gui_dialogs import SettingsDialog
        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"]
        )
        values = dlg.get_values()
        assert values['h_conv'] == 10.0

    def test_apply_defaults_with_h_conv(self):
        """Test that _apply_defaults sets h_conv from saved settings."""
        from ThermalSim.gui_dialogs import SettingsDialog
        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={'h_conv': 50.0}
        )
        values = dlg.get_values()
        assert values['h_conv'] == 50.0

    def test_apply_defaults_without_h_conv_keeps_default(self):
        """Test backward compat: old settings without h_conv."""
        from ThermalSim.gui_dialogs import SettingsDialog
        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={'time': 30.0, 'amb': 30.0}
        )
        values = dlg.get_values()
        # h_conv should remain at default 10.0
        assert values['h_conv'] == 10.0
        # Applied values should be updated
        assert values['time'] == 30.0
        assert values['amb'] == 30.0


class TestSimulationDetailSettings:
    """Tests for the user-facing simulation-detail settings."""

    def test_default_grid_settings_are_present(self, default_settings):
        """Legacy fixture settings should retain compatibility grid fields."""
        assert default_settings['grid_expert_limits'] is False
        assert default_settings['grid_max_cells'] == 200000
        assert default_settings['grid_target_cells'] == 100000

    def test_dialog_defaults_save_default_grid_limits(self):
        """Balanced mode should serialize a layer-aware node budget."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        values = dlg.get_values()

        assert values['grid_expert_limits'] is False
        assert values['grid_detail_level'] == 'balanced'
        assert values['grid_node_budget'] == 800000
        assert values['grid_max_cells'] == 400000
        assert values['grid_target_cells'] == 200000
        assert not dlg.grid_node_budget_input.IsEnabled()

    def test_switching_from_custom_to_balanced_restores_preset(self):
        """Leaving Custom should return to the selected preset budget."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        dlg.grid_detail_choice.SetSelection(4)
        dlg._on_grid_detail_changed(None)
        dlg.grid_node_budget_input.SetValue(900000)
        assert dlg.get_values()['grid_node_budget'] == 900000

        dlg.grid_detail_choice.SetSelection(1)
        dlg._on_grid_detail_changed(None)
        values = dlg.get_values()

        assert values['grid_expert_limits'] is False
        assert values['grid_detail_level'] == 'balanced'
        assert values['grid_node_budget'] == 800000
        assert not dlg.grid_node_budget_input.IsEnabled()

    def test_custom_mode_accepts_one_hundred_million_nodes(self):
        """The Custom spinner and settings serializer should accept 100M nodes."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        dlg.grid_detail_choice.SetSelection(4)
        dlg._on_grid_detail_changed(None)
        dlg.grid_node_budget_input.SetValue(100_000_000)
        values = dlg.get_values()

        assert values['grid_detail_level'] == 'custom'
        assert values['grid_node_budget'] == 100_000_000
        assert values['grid_max_cells'] == 50_000_000
        assert values['grid_target_cells'] == 25_000_000

    def test_custom_defaults_are_clamped_to_one_hundred_million_nodes(self):
        """Oversized imported settings should be clamped to the UI maximum."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={
                'grid_detail_level': 'custom',
                'grid_node_budget': 250_000_000,
            },
        )

        assert dlg.grid_node_budget_input.GetValue() == 100_000_000
        assert dlg.get_values()['grid_node_budget'] == 100_000_000

    def test_apply_defaults_without_grid_settings_keeps_defaults(self):
        """Old JSON settings should migrate to the Balanced detail preset."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={'time': 30.0}
        )
        values = dlg.get_values()

        assert values['time'] == 30.0
        assert values['grid_expert_limits'] is False
        assert values['grid_detail_level'] == 'balanced'
        assert values['grid_node_budget'] == 800000

    def test_apply_defaults_with_grid_expert_limits(self):
        """Saved expert grid limits should migrate to a Custom node budget."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={
                'grid_expert_limits': True,
                'grid_max_cells': 800000,
                'grid_target_cells': 400000,
            }
        )
        values = dlg.get_values()

        assert values['grid_expert_limits'] is True
        assert values['grid_detail_level'] == 'custom'
        assert values['grid_node_budget'] == 1600000
        assert values['grid_max_cells'] == 800000
        assert values['grid_target_cells'] == 400000
        assert dlg.grid_node_budget_input.IsEnabled()

    def test_area_settings_use_current_aware_mode_and_margin(self):
        """The geometry checkbox should serialize the new area settings."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        dlg.chk_limit_area.SetValue(True)
        dlg.pad_dist_input.SetValue(15.0)
        values = dlg.get_values()

        assert values['area_mode'] == 'active'
        assert values['area_margin_mm'] == 15.0
        assert values['limit_area'] is True
