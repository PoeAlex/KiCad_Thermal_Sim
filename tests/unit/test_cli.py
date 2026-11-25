"""Unit and end-to-end tests for the headless ThermalSim CLI and __init__ guard."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest

from ThermalSim import thermalsim_cli


@pytest.fixture(autouse=True)
def _preserve_headless_env_var():
    """Restore THERMALSIM_HEADLESS after each CLI test to avoid cross-test leakage."""
    orig = os.environ.get("THERMALSIM_HEADLESS")
    yield
    if orig is None:
        os.environ.pop("THERMALSIM_HEADLESS", None)
    else:
        os.environ["THERMALSIM_HEADLESS"] = orig


def _load_init_module(monkeypatch, argv0: str, headless_env=None, register_mock=None):
    """Execute __init__.py in an isolated module namespace to test CLI guard."""
    pkg_dir = Path(thermalsim_cli.__file__).resolve().parent
    init_path = pkg_dir / "__init__.py"
    mod_name = "_test_thermalsim_init_guard"

    if headless_env is None:
        monkeypatch.delenv("THERMALSIM_HEADLESS", raising=False)
    else:
        monkeypatch.setenv("THERMALSIM_HEADLESS", headless_env)

    monkeypatch.setattr(sys, "argv", [argv0, "inspect", "board.kicad_pcb"])

    fake_plugin_calls = []

    class _FakeThermalPlugin:
        def register(self):
            fake_plugin_calls.append("register")
            if register_mock is not None:
                register_mock()

    fake_thermal_plugin_mod = types.ModuleType(f"{mod_name}.thermal_plugin")
    fake_thermal_plugin_mod.ThermalPlugin = _FakeThermalPlugin
    monkeypatch.setitem(sys.modules, f"{mod_name}.thermal_plugin", fake_thermal_plugin_mod)

    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(init_path),
        submodule_search_locations=[str(pkg_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, module)
    spec.loader.exec_module(module)
    return module, fake_plugin_calls


class TestInitHeadlessGuard:
    """Tests for __init__.py CLI/headless registration guard."""

    def test_dash_m_invocation_skips_plugin_registration(self, monkeypatch):
        mod, calls = _load_init_module(monkeypatch, "-m", headless_env=None)
        assert mod._HEADLESS is True
        assert calls == []
        assert os.environ.get("THERMALSIM_HEADLESS") == "1"

    def test_main_py_invocation_skips_plugin_registration(self, monkeypatch):
        pkg_dir = Path(thermalsim_cli.__file__).resolve().parent
        mod, calls = _load_init_module(
            monkeypatch, str(pkg_dir / "__main__.py"), headless_env=None
        )
        assert mod._HEADLESS is True
        assert calls == []

    def test_thermalsim_cli_py_invocation_skips_plugin_registration(self, monkeypatch):
        pkg_dir = Path(thermalsim_cli.__file__).resolve().parent
        mod, calls = _load_init_module(
            monkeypatch, str(pkg_dir / "thermalsim_cli.py"), headless_env=None
        )
        assert mod._HEADLESS is True
        assert calls == []

    def test_gui_invocation_registers_plugin_when_not_headless(self, monkeypatch):
        mod, calls = _load_init_module(monkeypatch, "kicad", headless_env=None)
        assert mod._HEADLESS is False
        assert calls == ["register"]


class TestCliParsingAndConfigMerging:
    """Tests for CLI flag parsing, --pad/--limit-comp/--values, and --config merging."""

    def test_parse_pad_specs(self):
        assert thermalsim_cli._parse_pad_spec("U1.37=5W") == {
            "ref": "U1",
            "pad": "37",
            "name": "U1.37",
            "power": "5.0",
        }
        assert thermalsim_cli._parse_pad_spec("U1.37=5.25w") == {
            "ref": "U1",
            "pad": "37",
            "name": "U1.37",
            "power": "5.25",
        }
        assert thermalsim_cli._parse_pad_spec("U1.37=5.0") == {
            "ref": "U1",
            "pad": "37",
            "name": "U1.37",
            "power": "5.0",
        }
        assert thermalsim_cli._parse_pad_spec("U2.EP=/tmp/profile.csv") == {
            "ref": "U2",
            "pad": "EP",
            "name": "U2.EP",
            "power": "/tmp/profile.csv",
        }

    def test_parse_limit_comp_and_values(self):
        assert thermalsim_cli._parse_limit_comp_spec("C*=105") == ("C*", 105.0)
        assert thermalsim_cli._parse_limit_comp_spec("U1=150.5") == ("U1", 150.5)
        assert thermalsim_cli._parse_values_list("1, 3, 5, 10") == [1.0, 3.0, 5.0, 10.0]

    def test_config_merging_with_cli_overrides(self, tmp_path, monkeypatch, capsys):
        cfg_file = tmp_path / "job.json"
        cfg_file.write_text(
            json.dumps({
                "power_pads": [
                    {"ref": "U1", "pad": "37", "name": "U1.37", "power": "2.0"},
                    {"ref": "R1", "pad": "1", "name": "R1.1", "power": "0.5"},
                ],
                "amb": 25.0,
                "h_conv": 10.0,
                "limits": {
                    "max_board_c": 130.0,
                    "components": {"R1": 110.0},
                },
            }),
            encoding="utf-8",
        )

        captured_calls = []

        class _FakeApi:
            def validate_job(self, board, job_config):
                captured_calls.append((board, job_config))
                return {"schema_version": "1.0", "command": "validate", "valid": True}

        monkeypatch.setattr(thermalsim_cli, "headless_api", _FakeApi())

        rc = thermalsim_cli.main([
            "validate",
            "board.kicad_pcb",
            "--config", str(cfg_file),
            "--pad", "U1.37=5W",
            "--pad", "U2.1=1.5",
            "--steady-state",
            "--amb", "115",
            "--h-conv", "12.5",
            "--detail", "detailed",
            "--node-budget", "250000",
            "--mesh", "adaptive",
            "--engine", "fast_native",
            "--use-heatsink",
            "--sink-mode", "heatsink",
            "--heatsink-rth", "3.5",
            "--heatsink-mass", "80",
            "--heatsink-material", "copper",
            "--heatsink-cp", "385",
            "--pad-k", "4.0",
            "--pad-th", "0.5",
            "--contact-factor", "0.9",
            "--sink-temp", "45",
            "--limit-board", "160",
            "--limit-heatsink", "95",
            "--limit-comp", "C*=105",
            "--limit-comp", "U1=150",
            "--power-scale", "1.2",
            "--copper-thickness-scale", "0.8",
            "--refill-zones",
            "--pretty",
        ])

        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["valid"] is True
        assert "\n  " in out

        board, cfg = captured_calls[0]
        assert board == "board.kicad_pcb"
        assert cfg["power_pads"] == [
            {"ref": "U1", "pad": "37", "name": "U1.37", "power": "5.0"},
            {"ref": "R1", "pad": "1", "name": "R1.1", "power": "0.5"},
            {"ref": "U2", "pad": "1", "name": "U2.1", "power": "1.5"},
        ]
        assert cfg["steady_state"] is True
        assert cfg["amb"] == 115.0
        assert cfg["h_conv"] == 12.5
        assert cfg["grid_detail_level"] == "detailed"
        assert cfg["grid_node_budget"] == 250000
        assert cfg["mesh_mode"] == "adaptive"
        assert cfg["compute_engine"] == "fast_native"
        assert cfg["use_heatsink"] is True
        assert cfg["pad_sink_mode"] == "heatsink"
        assert cfg["heatsink_rth_kw"] == 3.5
        assert cfg["heatsink_mass_g"] == 80.0
        assert cfg["heatsink_material"] == "copper"
        assert cfg["heatsink_cp_j_kgk"] == 385.0
        assert cfg["pad_k"] == 4.0
        assert cfg["pad_th"] == 0.5
        assert cfg["pad_contact_factor"] == 0.9
        assert cfg["pad_sink_temp_c"] == 45.0
        assert cfg["pad_sink_temp_use_ambient"] is False
        assert cfg["limits"] == {
            "max_board_c": 160.0,
            "max_heatsink_c": 95.0,
            "components": {"R1": 110.0, "C*": 105.0, "U1": 150.0},
        }
        assert cfg["power_scale"] == 1.2
        assert cfg["copper_thickness_scale"] == 0.8
        assert cfg["refill_zones"] is True


class TestStdoutPurityAndExitCodes:
    """Tests for stdout/stderr separation and exit codes 0, 1, 2, 3."""

    def test_stdout_purity_redirects_noisy_prints_to_stderr(self, monkeypatch, capsys):
        class _NoisyApi:
            def run_headless_simulation(self, board, cfg, **kwargs):
                print("[ThermalSim] noisy solver output")
                print("another diagnostic line")
                return {
                    "schema_version": "1.0",
                    "command": "run",
                    "status": "ok",
                    "limits_check": {"enabled": False, "passed": True, "violations": []},
                }

        monkeypatch.setattr(thermalsim_cli, "headless_api", _NoisyApi())
        rc = thermalsim_cli.main(["run", "board.kicad_pcb", "--pad", "U1.1=1W", "--no-plots", "--no-html"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "[ThermalSim] noisy solver output" in captured.err
        assert "another diagnostic line" in captured.err
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert "[ThermalSim]" not in captured.out

    @pytest.mark.parametrize(
        "argv",
        [
            ["validate", "b.kicad_pcb", "--pad", "U1=5"],
            ["validate", "b.kicad_pcb", "--pad", "U1.37"],
            ["validate", "b.kicad_pcb", "--pad", ".37=5"],
            ["validate", "b.kicad_pcb", "--pad", "U1.37=abcW"],
            ["validate", "b.kicad_pcb", "--pad", "U1.37=nan"],
            ["validate", "b.kicad_pcb", "--limit-comp", "U1"],
            ["validate", "b.kicad_pcb", "--limit-comp", "U1=hot"],
            ["validate", "b.kicad_pcb", "--steady-state", "--transient"],
            ["validate", "b.kicad_pcb", "--config", "/nonexistent/missing_job.json"],
            ["sweep", "b.kicad_pcb", "--param", "h_conv", "--values", "1,,3"],
            ["sweep", "b.kicad_pcb", "--param", "h_conv", "--values", "1,bad"],
            ["sweep", "b.kicad_pcb"],
        ],
    )
    def test_bad_cli_syntax_returns_exit_2_with_json(self, argv, capsys):
        rc = thermalsim_cli.main(argv)
        captured = capsys.readouterr()
        assert rc == 2
        payload = json.loads(captured.out)
        assert payload["schema_version"] == "1.0"
        assert payload["status"] == "validation_error"
        assert payload["valid"] is False
        assert payload["errors"][0]["code"] == "CLI_ERROR"

    def test_validation_error_from_api_returns_exit_2(self, monkeypatch, capsys):
        class _InvalidApi:
            def validate_job(self, board, cfg):
                return {
                    "schema_version": "1.0",
                    "command": "validate",
                    "valid": False,
                    "errors": [{"code": "PAD_NOT_FOUND", "target": "U1.99"}],
                }

        monkeypatch.setattr(thermalsim_cli, "headless_api", _InvalidApi())
        rc = thermalsim_cli.main(["validate", "b.kicad_pcb", "--pad", "U1.99=1"])
        assert rc == 2
        assert json.loads(capsys.readouterr().out)["valid"] is False

    def test_limit_violation_returns_exit_3(self, monkeypatch, capsys):
        class _ViolationApi:
            def run_headless_simulation(self, board, cfg, **kwargs):
                return {
                    "schema_version": "1.0",
                    "command": "run",
                    "status": "limit_violation",
                    "limits_check": {
                        "enabled": True,
                        "passed": False,
                        "violations": [{"rule": "max_board_c", "limit_c": 100.0, "actual_c": 120.0}],
                    },
                }

        monkeypatch.setattr(thermalsim_cli, "headless_api", _ViolationApi())
        rc = thermalsim_cli.main(["run", "b.kicad_pcb", "--pad", "U1.1=5"])
        assert rc == 3
        assert json.loads(capsys.readouterr().out)["status"] == "limit_violation"

    def test_runtime_exception_returns_exit_1_with_json_and_stderr_traceback(self, monkeypatch, capsys):
        class _CrashingApi:
            def inspect_board(self, board):
                raise RuntimeError("boom during load")

        monkeypatch.setattr(thermalsim_cli, "headless_api", _CrashingApi())
        rc = thermalsim_cli.main(["inspect", "b.kicad_pcb"])
        captured = capsys.readouterr()

        assert rc == 1
        assert "RuntimeError: boom during load" in captured.err
        payload = json.loads(captured.out)
        assert payload["schema_version"] == "1.0"
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "RUNTIME_ERROR"
        assert payload["error"]["type"] == "RuntimeError"
        assert "boom during load" in payload["error"]["message"]

    def test_sweep_defaults_to_no_plots_and_respects_plots_flag(self, monkeypatch, capsys):
        calls = []

        class _SweepApi:
            def run_parameter_sweep(self, board, cfg, param, values, **kwargs):
                calls.append((param, values, kwargs))
                return {"schema_version": "1.0", "command": "sweep", "status": "ok", "runs": []}

        monkeypatch.setattr(thermalsim_cli, "headless_api", _SweepApi())
        rc1 = thermalsim_cli.main(["sweep", "b.kicad_pcb", "--param", "h_conv", "--values", "5,10"])
        capsys.readouterr()
        rc2 = thermalsim_cli.main(["sweep", "b.kicad_pcb", "--param", "h_conv", "--values", "5,10", "--plots"])
        capsys.readouterr()

        assert rc1 == 0
        assert rc2 == 0
        assert calls[0][2]["generate_plots"] is False
        assert calls[1][2]["generate_plots"] is True


class TestEndToEndSubcommandsWithHeadlessApi:
    """End-to-end tests calling all 5 subcommands against the real headless_api."""

    def test_all_five_subcommands_end_to_end(self, mock_board_2layer, tmp_path, monkeypatch, capsys):
        import pcbnew
        from ThermalSim import headless_api as real_headless_api

        monkeypatch.setattr(thermalsim_cli, "headless_api", None)
        monkeypatch.setattr(pcbnew, "LoadBoard", lambda path: mock_board_2layer, raising=False)
        monkeypatch.setattr(real_headless_api.pcbnew, "LoadBoard", lambda path: mock_board_2layer, raising=False)
        board_path = mock_board_2layer.GetFileName()

        # 1. inspect
        rc_inspect = thermalsim_cli.main(["inspect", board_path])
        out_inspect = json.loads(capsys.readouterr().out)
        assert rc_inspect == 0
        assert out_inspect["command"] == "inspect"
        assert out_inspect["stackup"]["layer_count"] == 2
        assert any(fp["ref"] == "U1" for fp in out_inspect["footprints"])

        # 2. validate (valid and invalid)
        rc_val_ok = thermalsim_cli.main(["validate", board_path, "--pad", "U1.1=2W", "--steady-state"])
        out_val_ok = json.loads(capsys.readouterr().out)
        assert rc_val_ok == 0
        assert out_val_ok["valid"] is True

        rc_val_bad = thermalsim_cli.main(["validate", board_path, "--pad", "U1.999=2W"])
        out_val_bad = json.loads(capsys.readouterr().out)
        assert rc_val_bad == 2
        assert out_val_bad["valid"] is False
        assert out_val_bad["errors"][0]["code"] == "PAD_NOT_FOUND"

        # 3. run (passing & limit violation)
        run_a_dir = tmp_path / "run_a"
        rc_run_a = thermalsim_cli.main([
            "run", board_path,
            "--pad", "U1.1=0.1W",
            "--steady-state",
            "--amb", "25",
            "--h-conv", "10",
            "--res", "1.0",
            "--limit-board", "500",
            "--no-plots", "--no-html",
            "--out", str(run_a_dir),
        ])
        out_run_a = json.loads(capsys.readouterr().out)
        assert rc_run_a == 0
        assert out_run_a["status"] == "ok"
        assert out_run_a["limits_check"]["passed"] is True
        assert (run_a_dir / "summary.json").is_file()

        run_b_dir = tmp_path / "run_b"
        rc_run_b = thermalsim_cli.main([
            "run", board_path,
            "--pad", "U1.1=0.1W",
            "--steady-state",
            "--amb", "25",
            "--h-conv", "5",
            "--res", "1.0",
            "--limit-board", "25.001",
            "--no-plots", "--no-html",
            "--out", str(run_b_dir),
        ])
        out_run_b = json.loads(capsys.readouterr().out)
        assert rc_run_b == 3
        assert out_run_b["status"] == "limit_violation"
        assert out_run_b["limits_check"]["passed"] is False

        # 4. sweep
        sweep_dir = tmp_path / "sweep"
        rc_sweep = thermalsim_cli.main([
            "sweep", board_path,
            "--param", "h_conv",
            "--values", "5,10,20",
            "--pad", "U1.1=0.1W",
            "--steady-state",
            "--amb", "25",
            "--res", "1.0",
            "--out", str(sweep_dir),
        ])
        out_sweep = json.loads(capsys.readouterr().out)
        assert rc_sweep == 0
        assert out_sweep["command"] == "sweep"
        assert len(out_sweep["runs"]) == 3
        assert (sweep_dir / "sweep_summary.json").is_file()

        # 5. compare
        rc_compare = thermalsim_cli.main(["compare", str(run_a_dir), str(run_b_dir)])
        out_compare = json.loads(capsys.readouterr().out)
        assert rc_compare == 0
        assert out_compare["command"] == "compare"
        assert out_compare["delta_max_temp_c"] > 0

    def test_v048_convection_and_via_cli_flags(self, monkeypatch, capsys):
        captured_calls = []

        class _FakeApi:
            def validate_job(self, board_path, job_config):
                captured_calls.append((board_path, job_config))
                return {
                    "schema_version": "1.0",
                    "command": "validate",
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                }

        monkeypatch.setattr(thermalsim_cli, "headless_api", _FakeApi())
        rc = thermalsim_cli.main([
            "validate", "board.kicad_pcb",
            "--pad", "U1.1=2W",
            "--h-conv", "15",
            "--h-conv-bottom", "0",
            "--via-plating-um", "35",
            "--via-fill", "copper",
        ])
        assert rc == 0
        _ = json.loads(capsys.readouterr().out)
        _, cfg = captured_calls[0]
        assert cfg["h_conv"] == 15.0
        assert cfg["h_conv_bottom"] == 0.0
        assert cfg["via_plating_um"] == 35.0
        assert cfg["via_fill"] == "copper"
