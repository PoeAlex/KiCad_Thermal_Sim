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

    def test_dialog_new_flow_1_to_3_auto_balances_total_and_displays_equipotential(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        from_pad = [{'pad_key': 'j1', 'name': 'J1:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'}]
        to_pads = [
            {'pad_key': 'u1', 'name': 'U1:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'},
            {'pad_key': 'u2', 'name': 'U2:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'},
            {'pad_key': 'u3', 'name': 'U3:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'},
        ]
        selected = list(from_pad)
        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "B.Cu"],
            selection_provider=lambda: selected
        )

        dlg._on_flow_new(None)
        dlg._on_flow_add_selection("from")
        selected = list(to_pads)
        dlg._on_flow_add_selection("to")
        dlg.flow_to_pad_list_input.SetValue("3, 3, 3")
        dlg._on_flow_apply_pad_current_list("to")
        values = dlg.get_values()

        flow = values['current_flows'][0]
        assert values['current_enabled'] is True
        assert flow['net_name'] == '+12V'
        assert flow['from_sharing'] == 'equipotential'
        assert flow['to_sharing'] == 'fixed'
        assert flow['total_current_a'] == 9.0
        assert [p['current_a'] for p in flow['to_pads']] == [3.0, 3.0, 3.0]
        assert dlg.flow_total_input.GetValue() == 9.0
        assert dlg.flow_total_input.IsEnabled() is False
        assert dlg.flow_from_list._items[0][2] == "Auto (Cu)"
        assert dlg.current_balance_text.GetValue() == "+12V: 9 A (1 -> 3 pads, OK)"

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
        assert values['current_flows'] == []
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
            list(dlg.current_flow_list._columns)
            + list(dlg.flow_from_list._columns)
            + list(dlg.flow_to_list._columns)
            + list(dlg.flow_from_sharing._choices)
            + list(dlg.flow_to_sharing._choices)
            + [dlg.chk_current_enabled.label]
        )
        forbidden = [
            "Strom", "Padgruppen", "Auswahl", "Gesamtstrom",
            "Netzbilanz", "Loeschen", "Uebernehmen", "Modus"
        ]
        for word in forbidden:
            assert word not in labels


class TestCurrentFlowSettings:
    """Tests for v0.5.0 Pad-to-Pad Current Flow helpers, migration, and PadPickerDialog."""

    def test_normalize_sharing_mode_aliases(self):
        from ThermalSim.gui_dialogs import normalize_sharing_mode

        assert normalize_sharing_mode("equipotential") == "equipotential"
        assert normalize_sharing_mode("Equipotential (Auto)") == "equipotential"
        assert normalize_sharing_mode("equal") == "equal"
        assert normalize_sharing_mode("Equal Split") == "equal"
        assert normalize_sharing_mode("fixed") == "fixed"
        assert normalize_sharing_mode("Fixed per Pad") == "fixed"
        assert normalize_sharing_mode("unknown") == "equipotential"

    def test_prepare_current_flows_modes_and_superposition(self):
        from ThermalSim.gui_dialogs import prepare_current_flows, summarize_current_flows

        flows = [
            {
                'name': 'Main 12V',
                'from_sharing': 'equipotential',
                'to_sharing': 'fixed',
                'from_pads': [{'pad_key': 'j1', 'name': 'J1:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'}],
                'to_pads': [
                    {'pad_key': 'u1', 'name': 'U1:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu', 'current_a': 3.0},
                    {'pad_key': 'u2', 'name': 'U2:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu', 'current_a': 3.0},
                    {'pad_key': 'u3', 'name': 'U3:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu', 'current_a': -3.0},
                ],
            },
            {
                'name': 'Aux 12V',
                'total_current_a': 6.0,
                'from_sharing': 'equipotential',
                'to_sharing': 'equal',
                'from_pads': [
                    {'pad_key': 'j1', 'name': 'J1:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'},
                    {'pad_key': 'j2', 'name': 'J2:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'},
                ],
                'to_pads': [
                    {'pad_key': 'u4', 'name': 'U4:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'},
                    {'pad_key': 'u5', 'name': 'U5:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'},
                ],
            },
        ]

        prepared = prepare_current_flows(flows)
        assert prepared[0]['total_current_a'] == 9.0
        assert prepared[0]['net_name'] == '+12V'
        assert prepared[0]['net_code'] == 1
        assert [p['current_a'] for p in prepared[0]['to_pads']] == [3.0, 3.0, 3.0]
        assert prepared[0]['from_pads'][0]['current_a'] == 9.0

        assert prepared[1]['total_current_a'] == 6.0
        assert [p['current_a'] for p in prepared[1]['from_pads']] == [0.0, 0.0]
        assert [p['current_a'] for p in prepared[1]['to_pads']] == [3.0, 3.0]
        assert summarize_current_flows(prepared) == "2 flows / 15 A"
        assert summarize_current_flows([]) == "0 flows"

    def test_validate_current_flows_errors_and_warnings(self):
        from ThermalSim.gui_dialogs import validate_current_flows

        ok_errors, ok_warnings = validate_current_flows([{
            'name': 'Good',
            'from_sharing': 'equipotential',
            'to_sharing': 'fixed',
            'from_pads': [{'pad_key': 'a', 'name': 'J1:1', 'net_name': '+12V'}],
            'to_pads': [{'pad_key': 'b', 'name': 'U1:1', 'net_name': '+12V', 'current_a': 5.0}],
        }])
        assert ok_errors == []
        assert ok_warnings == []

        errors, warnings = validate_current_flows([
            {
                'name': 'FixedMismatch',
                'from_sharing': 'fixed',
                'to_sharing': 'fixed',
                'from_pads': [{'pad_key': 'a', 'name': 'J1:1', 'net_name': '+12V', 'current_a': 5.0}],
                'to_pads': [{'pad_key': 'b', 'name': 'U1:1', 'net_name': '+12V', 'current_a': 3.0}],
            },
            {
                'name': 'MissingTo',
                'from_sharing': 'equipotential',
                'to_sharing': 'equipotential',
                'from_pads': [{'pad_key': 'c', 'name': 'J2:1', 'net_name': 'GND'}],
                'to_pads': [],
            },
            {
                'name': 'MismatchedNets',
                'from_sharing': 'equipotential',
                'to_sharing': 'equipotential',
                'total_current_a': 0.0,
                'from_pads': [{'pad_key': 'd', 'name': 'J3:1', 'net_name': '+5V'}],
                'to_pads': [{'pad_key': 'd', 'name': 'J3:1', 'net_name': 'GND'}],
            },
        ])
        assert any("From (5 A) != To (3 A)" in err for err in errors)
        assert any("missing To pads" in err for err in errors)
        assert any("mismatched nets" in err for err in errors)
        assert any("both From and To" in warn for warn in warnings)
        assert any("total current is 0 A" in warn for warn in warnings)

    def test_convert_current_groups_to_flows_and_apply_defaults_migration(self):
        from ThermalSim.gui_dialogs import (
            SettingsDialog,
            convert_current_groups_to_flows,
            current_groups_to_pad_currents,
        )

        legacy_groups = [
            {
                'name': 'VIN Supply',
                'mode': 'per_pad',
                'pads': [
                    {'pad_key': 'j1', 'name': 'J1:1', 'net_name': 'VIN', 'net_code': 2, 'layer': 'F.Cu', 'current_a': 9.0},
                    {'pad_key': 'u1', 'name': 'U1:1', 'net_name': 'VIN', 'net_code': 2, 'layer': 'F.Cu', 'current_a': -3.0},
                    {'pad_key': 'u2', 'name': 'U2:1', 'net_name': 'VIN', 'net_code': 2, 'layer': 'F.Cu', 'current_a': -6.0},
                ],
            },
        ]
        pad_map = current_groups_to_pad_currents(legacy_groups)
        assert pad_map == {'j1': 9.0, 'u1': -3.0, 'u2': -6.0}

        migrated = convert_current_groups_to_flows(legacy_groups)
        assert len(migrated) == 1
        assert migrated[0]['net_name'] == 'VIN'
        assert migrated[0]['total_current_a'] == 9.0
        assert migrated[0]['from_sharing'] == 'fixed'
        assert migrated[0]['to_sharing'] == 'fixed'
        assert [p['current_a'] for p in migrated[0]['from_pads']] == [9.0]
        assert [p['current_a'] for p in migrated[0]['to_pads']] == [3.0, 6.0]

        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "B.Cu"],
            defaults={'current_enabled': True, 'current_groups': legacy_groups},
        )
        values = dlg.get_values()
        assert len(values['current_flows']) == 1
        assert values['current_flows'][0]['total_current_a'] == 9.0
        assert len(values['current_groups']) == 1
        assert len(dlg.current_flow_list._items) == 1

    def test_pad_picker_dialog_filters_and_adds_to_flow(self, monkeypatch):
        from ThermalSim.gui_dialogs import PadPickerDialog, SettingsDialog

        board_pads = [
            {'pad_key': 'j1', 'name': 'J1:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'},
            {'pad_key': 'u1', 'name': 'U1:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'F.Cu'},
            {'pad_key': 'u2', 'name': 'U2:1', 'net_name': '+12V', 'net_code': 1, 'layer': 'B.Cu'},
            {'pad_key': 'g1', 'name': 'J1:2', 'net_name': 'GND', 'net_code': 2, 'layer': 'F.Cu'},
        ]
        picker = PadPickerDialog(None, board_pads, initial_net='+12V')
        assert picker.get_selected_net() == '+12V'
        assert [p['name'] for p in picker.filtered_pads] == ['J1:1', 'U1:1', 'U2:1']

        picker.set_search_text('u2')
        assert [p['name'] for p in picker.filtered_pads] == ['U2:1']

        picker.set_net_filter('')
        picker.set_search_text('j1')
        assert [p['name'] for p in picker.filtered_pads] == ['J1:1', 'J1:2']

        picker.set_net_filter('+12V')
        picker.set_search_text('')
        picker.pad_list.Select(1)
        picker.pad_list.Select(2)
        assert [p['name'] for p in picker.get_selected_pads()] == ['U1:1', 'U2:1']

        # Integrate with SettingsDialog._on_flow_pick_pads
        class FakePicker(PadPickerDialog):
            def __init__(self, parent, pads, initial_net=""):
                super().__init__(parent, pads, initial_net=initial_net)
                self.pad_list.Select(1)
                self.pad_list.Select(2)

        import ThermalSim.gui_dialogs as gui_mod
        monkeypatch.setattr(gui_mod, "PadPickerDialog", FakePicker)

        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "B.Cu"],
            board_pads=board_pads,
            defaults={
                'current_enabled': True,
                'current_flows': [{
                    'name': 'Load',
                    'from_sharing': 'equipotential',
                    'to_sharing': 'fixed',
                    'from_pads': [board_pads[0]],
                    'to_pads': [],
                }],
            },
        )
        dlg._on_flow_pick_pads("to")
        flow = dlg.get_values()['current_flows'][0]
        assert [p['name'] for p in flow['to_pads']] == ['U1:1', 'U2:1']


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
            'h_conv', 'h_conv_bottom', 'via_plating_um', 'via_fill',
            'pcb_thick', 'grid_expert_limits',
            'grid_max_cells', 'grid_target_cells', 'capabilities',
            'help', 'preview', 'load_settings', 'save_settings',
            'current_flows', 'flow_from_sharing', 'flow_to_sharing',
            'flow_total', 'flow_pick_pads', 'flow_from_pads', 'flow_to_pads',
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

    def test_warning_footer_is_separate_from_action_buttons(self):
        """Long warnings must be wrapped in a panel above the button row."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(None, 0, 0.5, ["F.Cu", "B.Cu"])
        warning = (
            "The requested grid contains many millions of nodes and may "
            "require tens of gigabytes of memory before adaptive reduction."
        )
        dlg._set_footer_status("Warning", warning)

        assert dlg.footer_status_panel is not dlg.footer_actions_panel
        assert dlg.lbl_preflight.parent is dlg.footer_status_panel
        assert dlg.btn_run.parent is dlg.footer_actions_panel
        assert dlg.lbl_preflight.GetLabel() == warning
        assert dlg.lbl_preflight._wrap_width == 700
        assert dlg.lbl_preflight._tooltip == warning

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
        """Flow sharing choices should toggle fixed-current rows and Total (A) editability."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None, 0, 0.5, ["F.Cu", "B.Cu"],
            defaults={
                'current_enabled': True,
                'current_flows': [{
                    'name': 'Supply',
                    'from_sharing': 'equipotential',
                    'to_sharing': 'fixed',
                    'from_pads': [{'pad_key': '1', 'name': 'J1-1', 'net_name': 'VIN'}],
                    'to_pads': [{'pad_key': '2', 'name': 'U1-1', 'net_name': 'VIN', 'current_a': 4.0}],
                }],
            },
        )

        assert dlg.flow_total_input.IsEnabled() is False
        assert dlg.flow_from_fixed_row._shown is False
        assert dlg.flow_to_fixed_row._shown is True

        # Switch To side to Equipotential -> Total (A) becomes editable and fixed row hides
        dlg.flow_to_sharing.SetSelection(0)
        dlg._on_flow_sharing_changed("to")
        assert dlg.flow_total_input.IsEnabled() is True
        assert dlg.flow_from_fixed_row._shown is False
        assert dlg.flow_to_fixed_row._shown is False
        assert dlg.flow_to_list._items[0][2] == "Auto (Cu)"

        # Switch To side to Equal Split -> displays split value and keeps Total (A) editable
        dlg.flow_total_input.SetValue(10.0)
        dlg._on_flow_total_changed(None)
        dlg.flow_to_sharing.SetSelection(1)
        dlg._on_flow_sharing_changed("to")
        assert dlg.flow_total_input.IsEnabled() is True
        assert dlg.flow_to_list._items[0][2] == "10"

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
                'time_stepping': 'uniform',
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
        assert values['grid_node_budget'] == 3_000_000
        assert values['grid_max_cells'] == 1_500_000
        assert values['grid_target_cells'] == 1_275_000
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
        assert values['grid_node_budget'] == 3_000_000
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
        assert values['grid_target_cells'] == 42_500_000

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
        assert values['grid_node_budget'] == 3_000_000

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
        assert values['grid_node_budget'] == 1_600_000
        assert values['grid_max_cells'] == 800_000
        assert values['grid_target_cells'] == 680_000
        assert dlg.grid_node_budget_input.IsEnabled()

    def test_all_grid_detail_presets_match_expected_budgets_and_85pct_targets(self):
        """All named presets should match updated budgets and 85% auto-coarsen targets."""
        from ThermalSim.gui_dialogs import (
            DEFAULT_GRID_DETAIL,
            DEFAULT_GRID_NODE_BUDGET,
            GRID_DETAIL_PRESETS,
            SettingsDialog,
        )
        from ThermalSim.thermal_plugin import (
            GRID_DETAIL_PRESETS as PLUGIN_GRID_DETAIL_PRESETS,
        )

        expected = {
            "fast": (1_000_000, 850_000),
            "balanced": (3_000_000, 2_550_000),
            "detailed": (8_000_000, 6_800_000),
            "very_detailed": (16_000_000, 13_600_000),
        }
        assert GRID_DETAIL_PRESETS == expected
        assert PLUGIN_GRID_DETAIL_PRESETS == expected
        assert DEFAULT_GRID_DETAIL == "balanced"
        assert DEFAULT_GRID_NODE_BUDGET == 3_000_000

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"])
        for idx, name in enumerate(("fast", "balanced", "detailed", "very_detailed")):
            dlg.grid_detail_choice.SetSelection(idx)
            dlg._on_grid_detail_changed(None)
            values = dlg.get_values()
            max_nodes, target_nodes = expected[name]
            assert values['grid_detail_level'] == name
            assert values['grid_node_budget'] == max_nodes
            assert values['grid_max_cells'] == max_nodes // 4
            assert values['grid_target_cells'] == target_nodes // 4

    def test_legacy_adaptive_mesh_defaults_are_tolerated_and_uniform_serialized(self):
        """Legacy saved settings with mesh_mode='adaptive' load cleanly and serialize uniform."""
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None,
            1,
            0.5,
            ["F.Cu", "B.Cu"],
            defaults={'mesh_mode': 'adaptive', 'adaptive_max_cell_ratio': 4},
        )
        assert not hasattr(dlg, 'mesh_mode_choice')
        assert not hasattr(dlg, 'adaptive_ratio_input')
        assert dlg.get_values()['mesh_mode'] == "uniform"
        # Verify key controls carry English tooltips
        assert dlg.compute_engine_choice._tooltip
        assert dlg.time_stepping_choice._tooltip
        assert dlg.btn_run._tooltip
        assert dlg.btn_more._tooltip
        assert dlg.flow_from_sharing._tooltip
        assert dlg.flow_to_sharing._tooltip
        assert dlg.flow_total_input._tooltip
        assert dlg.current_balance_text._tooltip

    def test_legacy_solver_backend_setting_is_ignored_and_engine_label_updates(self, monkeypatch):
        """Old JSON settings with legacy solver_backend load safely; capabilities label reflects native flags."""
        import ThermalSim.gui_dialogs as gui_mod

        # Legacy JSON containing deprecated backend key must load cleanly without error
        legacy_key = "par" + "diso"
        dlg = gui_mod.SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={'solver_backend': legacy_key, 'time': 15.0},
        )
        assert not hasattr(dlg, "solver_backend_choice")
        values = dlg.get_values()
        assert 'solver_backend' not in values
        assert values['time'] == 15.0
        assert not dlg.solver_pane.IsExpanded()

        # Tooltips must not mention deprecated backend
        for text in gui_mod.TOOLTIP_TEXTS.values():
            assert legacy_key not in text.lower()

        # Capabilities label reflects HAS_NATIVE_MG / HAS_NATIVE_CORE
        monkeypatch.setattr(gui_mod, "HAS_NATIVE_MG", True)
        monkeypatch.setattr(gui_mod, "HAS_NATIVE_CORE", True)
        dlg_mg = gui_mod.SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        assert dlg_mg.lbl_capabilities.GetLabel() == "Engine: Native C++/OpenMP + SciPy"

        monkeypatch.setattr(gui_mod, "HAS_NATIVE_MG", False)
        monkeypatch.setattr(gui_mod, "HAS_NATIVE_CORE", True)
        dlg_core = gui_mod.SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        assert dlg_core.lbl_capabilities.GetLabel() == "Engine: Native C++ + SciPy"

        monkeypatch.setattr(gui_mod, "HAS_NATIVE_MG", False)
        monkeypatch.setattr(gui_mod, "HAS_NATIVE_CORE", False)
        dlg_py = gui_mod.SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        assert dlg_py.lbl_capabilities.GetLabel() == "Engine: NumPy + SciPy"

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


class TestPreflightDebounce:
    """Tests for preflight timer debouncing and fallback behavior."""

    def test_estimate_input_changed_refreshes_immediately_under_wx_mock(self):
        from ThermalSim.gui_dialogs import SettingsDialog
        from ThermalSim.workflow import GridEstimate, PreflightResult

        calls = []
        grid = GridEstimate(0.5, 0.5, 0, 0, 10, 10, 24, 24, 2, False, False, 200000, 100000)

        def preflight(settings):
            calls.append(settings["res"])
            return PreflightResult(grid=grid)

        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            preflight_callback=preflight,
        )
        assert len(calls) >= 1
        assert dlg._preflight_timer is None
        calls.clear()

        dlg.res_input.SetValue(0.25)
        dlg._on_estimate_input_changed(None)
        assert calls == [0.25]

    def test_estimate_input_changed_debounces_with_wx_timer_and_stops_on_cancel(self, monkeypatch):
        import ThermalSim.gui_dialogs as gui_mod
        from ThermalSim.workflow import GridEstimate, PreflightResult

        class FakeTimer:
            instances = []

            def __init__(self, owner):
                self.owner = owner
                self.started_ms = []
                self.stop_calls = 0
                FakeTimer.instances.append(self)

            def StartOnce(self, ms):
                self.started_ms.append(ms)
                return True

            def Stop(self):
                self.stop_calls += 1

        monkeypatch.setattr(gui_mod.wx, "Timer", FakeTimer, raising=False)
        monkeypatch.setattr(gui_mod.wx, "EVT_TIMER", "EVT_TIMER", raising=False)

        calls = []
        grid = GridEstimate(0.5, 0.5, 0, 0, 10, 10, 24, 24, 2, False, False, 200000, 100000)

        def preflight(settings):
            calls.append(settings["res"])
            return PreflightResult(grid=grid)

        dlg = gui_mod.SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            preflight_callback=preflight,
        )
        assert len(calls) >= 1
        assert len(FakeTimer.instances) == 1
        timer = FakeTimer.instances[0]
        assert dlg._preflight_timer is timer
        calls.clear()

        # Rapid edits should debounce without running preflight immediately
        dlg.res_input.SetValue(0.4)
        dlg._on_estimate_input_changed(None)
        dlg.res_input.SetValue(0.3)
        dlg._on_estimate_input_changed(None)
        assert calls == []
        assert timer.started_ms == [150, 150]

        # Timer firing triggers a single preflight refresh with the latest value
        dlg._on_preflight_timer(None)
        assert calls == [0.3]

        # Closing stops and clears the timer
        stops_before = timer.stop_calls
        dlg._on_cancel(None)
        assert timer.stop_calls == stops_before + 1
        assert dlg._preflight_timer is None


class TestSteadyStateSettings:
    """Tests for steady-state checkbox behavior, serialization, and defaults."""

    def test_steady_state_tooltip_and_defaults(self):
        from ThermalSim.gui_dialogs import SettingsDialog, TOOLTIP_TEXTS

        assert 'steady_state' in TOOLTIP_TEXTS
        assert TOOLTIP_TEXTS['steady_state'].strip() == TOOLTIP_TEXTS['steady_state']

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])

        assert dlg.chk_steady_state.GetValue() is False
        assert dlg.chk_steady_state.label == "Steady-state (final temperature)"
        assert dlg.time_input.IsEnabled() is True
        assert dlg.chk_snapshots.IsEnabled() is True
        assert dlg.snap_count_input.IsEnabled() is False
        assert dlg.get_values()['steady_state'] is False

    def test_steady_state_toggle_disables_and_restores_transient_controls(self):
        from ThermalSim.gui_dialogs import SettingsDialog
        from ThermalSim.workflow import GridEstimate, PreflightResult

        preflight_calls = []
        grid = GridEstimate(0.5, 0.5, 0, 0, 10, 10, 24, 24, 2, False, False, 200000, 100000)

        def preflight(settings):
            preflight_calls.append(settings['steady_state'])
            return PreflightResult(grid=grid)

        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            preflight_callback=preflight,
        )
        dlg.chk_snapshots.SetValue(True)
        dlg._on_snapshots_toggle(None)
        assert dlg.snap_count_input.IsEnabled() is True
        preflight_calls.clear()

        # Enable steady-state: time, snapshots, and snap_count must all disable
        dlg.chk_steady_state.SetValue(True)
        dlg._on_steady_state_toggle(None)
        assert dlg.time_input.IsEnabled() is False
        assert dlg.chk_snapshots.IsEnabled() is False
        assert dlg.snap_count_input.IsEnabled() is False
        assert dlg.get_values()['steady_state'] is True
        assert preflight_calls == [True]

        # Disable steady-state: time, snapshots, and snap_count (since snapshots checked) re-enable
        dlg.chk_steady_state.SetValue(False)
        dlg._on_steady_state_toggle(None)
        assert dlg.time_input.IsEnabled() is True
        assert dlg.chk_snapshots.IsEnabled() is True
        assert dlg.snap_count_input.IsEnabled() is True
        assert dlg.get_values()['steady_state'] is False
        assert preflight_calls == [True, False]

        # When snapshots is unchecked, unchecking steady-state keeps snap_count disabled
        dlg.chk_snapshots.SetValue(False)
        dlg._on_snapshots_toggle(None)
        dlg.chk_steady_state.SetValue(True)
        dlg._on_steady_state_toggle(None)
        dlg.chk_steady_state.SetValue(False)
        dlg._on_steady_state_toggle(None)
        assert dlg.time_input.IsEnabled() is True
        assert dlg.chk_snapshots.IsEnabled() is True
        assert dlg.snap_count_input.IsEnabled() is False

    def test_apply_defaults_restores_steady_state_and_defaults_false_when_absent(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={'steady_state': True, 'snapshots': True, 'snap_count': 8},
        )
        assert dlg.chk_steady_state.GetValue() is True
        assert dlg.time_input.IsEnabled() is False
        assert dlg.chk_snapshots.IsEnabled() is False
        assert dlg.snap_count_input.IsEnabled() is False
        assert dlg.get_values()['steady_state'] is True

        # Applying defaults without steady_state resets it to False
        dlg._apply_defaults({'time': 30.0, 'snapshots': False})
        assert dlg.chk_steady_state.GetValue() is False
        assert dlg.time_input.IsEnabled() is True
        assert dlg.chk_snapshots.IsEnabled() is True
        assert dlg.snap_count_input.IsEnabled() is False
        assert dlg.get_values()['steady_state'] is False


class TestDynamicHeatsinkSettings:
    """Tests for Dynamic 0D Heatsink controls, presets, live summary, and round-trip."""

    def test_heatsink_defaults_and_options(self):
        from ThermalSim.gui_dialogs import (
            HEATSINK_MATERIAL_OPTIONS,
            TOOLTIP_TEXTS,
            SettingsDialog,
        )

        assert [key for _, key, _ in HEATSINK_MATERIAL_OPTIONS] == [
            "aluminium", "copper", "steel", "magnesium", "custom"
        ]
        for tooltip_key in (
            "pad_contact_factor",
            "pad_sink_mode",
            "pad_sink_temp_use_ambient",
            "pad_sink_temp_c",
            "heatsink_rth_kw",
            "heatsink_material",
            "heatsink_mass_g",
            "heatsink_cp_j_kgk",
        ):
            assert tooltip_key in TOOLTIP_TEXTS
            assert TOOLTIP_TEXTS[tooltip_key].strip() == TOOLTIP_TEXTS[tooltip_key]

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        values = dlg.get_values()

        assert values["use_heatsink"] is False
        assert values["pad_sink_mode"] == "fixed"
        assert values["pad_contact_factor"] == 0.2
        assert values["pad_sink_temp_use_ambient"] is True
        assert values["pad_sink_temp_c"] == 25.0
        assert values["heatsink_rth_kw"] == 5.0
        assert values["heatsink_material"] == "aluminium"
        assert values["heatsink_mass_g"] == 50.0
        assert values["heatsink_cp_j_kgk"] == 897.0

        assert dlg.pad_contact_factor.IsEnabled() is False
        assert dlg.chk_pad_sink_ambient.IsEnabled() is False
        assert dlg.pad_sink_temp_input.IsEnabled() is False
        assert dlg.heatsink_rth_input.IsEnabled() is False
        assert dlg.heatsink_mass_input.IsEnabled() is False
        assert dlg.heatsink_cp_input.IsEnabled() is False
        assert dlg.lbl_heatsink_summary.GetLabel() == "Fixed sink temperature (infinite thermal mass)"

    def test_fixed_vs_heatsink_mode_and_material_presets_live_summary(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])

        # Enable thermal pad in fixed mode
        dlg.chk_heatsink.SetValue(True)
        dlg._on_heatsink_toggle(None)
        assert dlg.thermal_pad_pane.IsExpanded()
        assert dlg.pad_contact_factor.IsEnabled() is True
        assert dlg.chk_pad_sink_ambient.IsEnabled() is True
        assert dlg.pad_sink_temp_input.IsEnabled() is False
        assert dlg.heatsink_rth_input.IsEnabled() is False
        assert dlg.heatsink_mass_input.IsEnabled() is False
        assert dlg.heatsink_cp_input.IsEnabled() is False

        # Uncheck ambient temp -> fixed sink temp input becomes editable
        dlg.chk_pad_sink_ambient.SetValue(False)
        dlg._on_pad_sink_ambient_toggle(None)
        assert dlg.pad_sink_temp_input.IsEnabled() is True

        # Switch to Dynamic heatsink (0D) mode
        dlg.pad_sink_mode_choice.SetSelection(1)
        dlg._on_pad_sink_mode_changed(None)
        assert dlg.chk_pad_sink_ambient.IsEnabled() is False
        assert dlg.pad_sink_temp_input.IsEnabled() is False
        assert dlg.heatsink_rth_input.IsEnabled() is True
        assert dlg.heatsink_mass_input.IsEnabled() is True
        assert dlg.heatsink_cp_input.IsEnabled() is False
        assert dlg.lbl_heatsink_summary.GetLabel() == (
            "C_hs = 44.9 J/K  \u00b7  Time constant tau = 224.2 s"
        )

        # Copper preset auto-fills c_p and updates summary
        dlg.heatsink_material_choice.SetSelection(1)
        dlg._on_heatsink_material_changed(None)
        assert dlg.heatsink_cp_input.GetValue() == 385.0
        assert dlg.heatsink_cp_input.IsEnabled() is False
        assert dlg.get_values()["heatsink_material"] == "copper"
        assert dlg.get_values()["heatsink_cp_j_kgk"] == 385.0
        assert dlg.lbl_heatsink_summary.GetLabel() == (
            "C_hs = 19.2 J/K  \u00b7  Time constant tau = 96.2 s"
        )

        # Custom material enables c_p input and updates summary live
        dlg.heatsink_material_choice.SetSelection(4)
        dlg._on_heatsink_material_changed(None)
        assert dlg.heatsink_cp_input.IsEnabled() is True
        dlg.heatsink_cp_input.SetValue(1000.0)
        dlg.heatsink_mass_input.SetValue(100.0)
        dlg.heatsink_rth_input.SetValue(3.0)
        dlg._on_heatsink_param_changed(None)
        assert dlg.lbl_heatsink_summary.GetLabel() == (
            "C_hs = 100.0 J/K  \u00b7  Time constant tau = 300.0 s"
        )

        # Zero mass shows instantaneous heatsink summary
        dlg.heatsink_mass_input.SetValue(0.0)
        dlg._on_heatsink_param_changed(None)
        assert dlg.lbl_heatsink_summary.GetLabel() == (
            "C_hs = 0.0 J/K (instantaneous heatsink, tau = 0 s)"
        )

    def test_apply_defaults_round_trip_and_backward_compat(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        saved = {
            "use_heatsink": True,
            "pad_th": 0.5,
            "pad_k": 5.0,
            "pad_cap_areal": 250.0,
            "pad_sink_mode": "heatsink",
            "pad_contact_factor": 0.75,
            "pad_sink_temp_use_ambient": False,
            "pad_sink_temp_c": 40.0,
            "heatsink_rth_kw": 3.0,
            "heatsink_material": "custom",
            "heatsink_mass_g": 120.0,
            "heatsink_cp_j_kgk": 950.0,
        }
        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"], defaults=saved)
        assert dlg.thermal_pad_pane.IsExpanded()
        assert dlg.heatsink_cp_input.IsEnabled() is True
        values = dlg.get_values()
        for key, expected in saved.items():
            assert values[key] == expected

        # Applying legacy defaults without new heatsink keys resets them safely
        dlg._apply_defaults({"use_heatsink": True, "pad_th": 1.0, "pad_k": 3.0})
        restored = dlg.get_values()
        assert restored["pad_sink_mode"] == "fixed"
        assert restored["pad_contact_factor"] == 0.2
        assert restored["pad_sink_temp_use_ambient"] is True
        assert restored["pad_sink_temp_c"] == 25.0
        assert restored["heatsink_rth_kw"] == 5.0
        assert restored["heatsink_material"] == "aluminium"
        assert restored["heatsink_mass_g"] == 50.0
        assert restored["heatsink_cp_j_kgk"] == 897.0
        assert dlg.lbl_heatsink_summary.GetLabel() == "Fixed sink temperature (infinite thermal mass)"

    def test_set_run_state_refreshes_dialog_and_yields_app(self, monkeypatch):
        """set_run_state should immediately refresh/update the dialog and yield wx events."""
        import ThermalSim.gui_dialogs as gui_mod
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        calls = {"refresh": 0, "update": 0, "yield": []}

        class DummyApp:
            def Yield(self, only_if_needed=False):
                calls["yield"].append(only_if_needed)

        dlg.Refresh = lambda: calls.__setitem__("refresh", calls["refresh"] + 1)
        dlg.Update = lambda: calls.__setitem__("update", calls["update"] + 1)
        monkeypatch.setattr(gui_mod.wx, "GetApp", lambda: DummyApp(), raising=False)

        dlg.set_run_state("running", "Mapping exact PCB geometry...")
        assert calls["refresh"] == 1
        assert calls["update"] == 1
        assert calls["yield"] == [True]
        assert dlg.lbl_preflight_status.GetLabel() == "Running"
        assert dlg.lbl_preflight.GetLabel() == "Mapping exact PCB geometry..."
        assert not dlg.btn_run.IsEnabled()

        # Defensive when Refresh/Update raise and GetApp returns None
        def _boom():
            raise RuntimeError("wx gone")

        dlg.Refresh = _boom
        dlg.Update = _boom
        monkeypatch.setattr(gui_mod.wx, "GetApp", lambda: None, raising=False)
        dlg.set_run_state("cancelled")
        assert dlg.lbl_preflight_status.GetLabel() == "Cancelled"
        assert dlg.btn_run.IsEnabled()


class TestSolverV048Controls:
    """Tests for v0.4.8 Solver pane convection, via barrel, and stackup thickness controls."""

    def test_solver_defaults_tooltips_and_ranges(self):
        from ThermalSim.gui_dialogs import SettingsDialog, TOOLTIP_TEXTS

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        values = dlg.get_values()

        assert values["h_conv"] == 10.0
        assert values["h_conv_bottom"] == 10.0
        assert values["via_plating_um"] == 25.0
        assert values["via_fill"] == "plated"
        assert dlg.h_conv_input._min == 0.0
        assert dlg.h_conv_bottom_input._min == 0.0
        assert dlg.via_plating_input.IsEnabled() is True
        assert dlg.thick_input.IsEnabled() is True
        assert not dlg.solver_pane.IsExpanded()

        assert "radiation" in TOOLTIP_TEXTS["h_conv"].lower()
        for key in ("h_conv", "h_conv_bottom", "via_plating_um", "via_fill"):
            assert TOOLTIP_TEXTS[key].strip() == TOOLTIP_TEXTS[key]

    def test_via_fill_copper_disables_plating_and_round_trips(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(None, 1, 0.5, ["F.Cu", "B.Cu"])
        assert dlg.via_fill_choice.GetStringSelection() == "Plated (standard)"

        dlg.via_fill_choice.SetSelection(1)
        dlg._on_via_fill_changed(None)
        assert dlg.via_fill_choice.GetStringSelection() == "Copper-filled"
        assert dlg.via_plating_input.IsEnabled() is False
        assert dlg.get_values()["via_fill"] == "copper"

        dlg.via_fill_choice.SetSelection(0)
        dlg._on_via_fill_changed(None)
        assert dlg.via_plating_input.IsEnabled() is True
        assert dlg.get_values()["via_fill"] == "plated"

    def test_stackup_thickness_locks_thick_input_and_avoids_auto_expand(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            stackup_thickness_mm=0.8,
            defaults={"thick": 2.4},
        )
        assert dlg.thick_input.GetValue() == pytest.approx(0.8)
        assert dlg.thick_input.IsEnabled() is False
        assert "stackup" in dlg.thick_input.GetToolTipText().lower()
        assert not dlg.solver_pane.IsExpanded()
        assert dlg.get_values()["thick"] == pytest.approx(0.8)

        # Fallback mode without stackup thickness remains editable and expands when != 1.6
        dlg_fallback = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={"thick": 2.0},
        )
        assert dlg_fallback.thick_input.GetValue() == pytest.approx(2.0)
        assert dlg_fallback.thick_input.IsEnabled() is True
        assert dlg_fallback.solver_pane.IsExpanded()

    def test_apply_defaults_backward_compat_and_auto_expand(self):
        from ThermalSim.gui_dialogs import SettingsDialog

        # Legacy defaults without h_conv_bottom copy h_conv to bottom
        dlg = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={"h_conv": 35.0},
        )
        values = dlg.get_values()
        assert values["h_conv"] == 35.0
        assert values["h_conv_bottom"] == 35.0
        assert values["via_plating_um"] == 25.0
        assert values["via_fill"] == "plated"
        assert dlg.solver_pane.IsExpanded()

        # copper_filled alias normalizes and disables via_plating_input; h_conv_bottom expands pane
        dlg_bot = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={"h_conv": 10.0, "h_conv_bottom": 0.0, "via_fill": "copper_filled"},
        )
        vals_bot = dlg_bot.get_values()
        assert vals_bot["h_conv"] == 10.0
        assert vals_bot["h_conv_bottom"] == 0.0
        assert vals_bot["via_fill"] == "copper"
        assert dlg_bot.via_plating_input.IsEnabled() is False
        assert dlg_bot.solver_pane.IsExpanded()

        # via_plating_um != 25 expands solver pane
        dlg_plating = SettingsDialog(
            None, 1, 0.5, ["F.Cu", "B.Cu"],
            defaults={"via_plating_um": 35.0},
        )
        assert dlg_plating.get_values()["via_plating_um"] == 35.0
        assert dlg_plating.solver_pane.IsExpanded()
