"""Unit tests for ThermalSim.headless_api."""

import ast
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ThermalSim import headless_api
from ThermalSim.thermal_solver import SolverResult
from tests.mocks.pcbnew_mock import (
    B_Cu,
    EDA_RECT,
    Eco1_User,
    F_Cu,
    In1_Cu,
    In2_Cu,
    MockBoard,
    MockDrawing,
    MockFootprint,
    MockPad,
    MockTrack,
    MockZone,
    PAD_ATTRIB_PTH,
    PAD_ATTRIB_SMD,
    VECTOR2I,
)


def _make_board(
    sample_2layer_board_file=None,
    *,
    with_eco1=True,
    unfilled_zone=False,
    extra_fps=None,
):
    """Build a deterministic test MockBoard with multiple footprints and pads."""
    u1_pad1 = MockPad(
        position=VECTOR2I(10_000_000, 10_000_000),
        layer=F_Cu,
        bbox=EDA_RECT(9_000_000, 9_000_000, 2_000_000, 2_000_000),
        net_code=1,
        net_name="VIN",
        number="1",
    )
    u1_ep = MockPad(
        position=VECTOR2I(12_000_000, 10_000_000),
        layer=F_Cu,
        bbox=EDA_RECT(10_500_000, 8_500_000, 3_000_000, 3_000_000),
        net_code=2,
        net_name="GND",
        number="EP",
    )
    c1_pad1 = MockPad(
        position=VECTOR2I(35_000_000, 35_000_000),
        layer=F_Cu,
        bbox=EDA_RECT(34_500_000, 34_500_000, 1_000_000, 1_000_000),
        net_code=1,
        net_name="VIN",
        number="1",
    )
    c1_pad2 = MockPad(
        position=VECTOR2I(37_000_000, 35_000_000),
        layer=F_Cu,
        bbox=EDA_RECT(36_500_000, 34_500_000, 1_000_000, 1_000_000),
        net_code=2,
        net_name="GND",
        number="2",
    )

    fps = [
        MockFootprint(reference="U1", pads=[u1_pad1, u1_ep]),
        MockFootprint(reference="C1", pads=[c1_pad1, c1_pad2]),
    ]
    if extra_fps:
        fps.extend(extra_fps)

    tracks = [
        MockTrack(
            layer=F_Cu,
            bbox=EDA_RECT(10_000_000, 9_500_000, 25_000_000, 1_000_000),
            start=VECTOR2I(10_000_000, 10_000_000),
            end=VECTOR2I(35_000_000, 10_000_000),
            width=1_000_000,
            net_code=1,
            net_name="VIN",
        ),
        MockTrack(
            layer=F_Cu,
            bbox=EDA_RECT(34_500_000, 10_000_000, 1_000_000, 25_000_000),
            start=VECTOR2I(35_000_000, 10_000_000),
            end=VECTOR2I(35_000_000, 35_000_000),
            width=1_000_000,
            net_code=1,
            net_name="VIN",
        ),
    ]
    zones = [
        MockZone(
            layers=[F_Cu, B_Cu],
            bbox=EDA_RECT(0, 0, 50_000_000, 50_000_000),
            filled=not unfilled_zone,
            net_code=2,
            net_name="GND",
        )
    ]
    drawings = (
        [MockDrawing(layer=Eco1_User, bbox=EDA_RECT(8_000_000, 8_000_000, 8_000_000, 8_000_000))]
        if with_eco1
        else []
    )

    return MockBoard(
        filename=sample_2layer_board_file or "",
        footprints=fps,
        tracks=tracks,
        zones=zones,
        drawings=drawings,
        layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu", Eco1_User: "User.Eco1"},
    )


class TestHeadlessIsolationAndJson:
    """Verify zero wx imports and strict JSON serializability."""

    def test_constants(self):
        assert headless_api.SCHEMA_VERSION == "1.0"
        assert headless_api.PLUGIN_VERSION == "0.4.4"

    def test_no_wx_or_gui_imports_in_ast(self):
        path = os.path.abspath(headless_api.__file__)
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lstrip(".").split(".")[0])

        assert "wx" not in imported
        assert "thermal_plugin" not in imported
        assert "gui_dialogs" not in imported

    def test_fresh_subprocess_import_without_wx(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        parent_dir = os.path.dirname(project_root)
        pkg_name = os.path.basename(project_root)
        code = (
            "import sys, types\n"
            "sys.modules['wx'] = None\n"
            f"sys.path.insert(0, {os.path.join(project_root, 'tests')!r})\n"
            "from mocks.pcbnew_mock import install_mock\n"
            "install_mock()\n"
            f"sys.path.insert(0, {parent_dir!r})\n"
            f"mod = __import__('{pkg_name}.headless_api', fromlist=['SCHEMA_VERSION'])\n"
            "assert mod.SCHEMA_VERSION == '1.0'\n"
            "assert 'thermal_plugin' not in str(list(sys.modules.keys()))\n"
            "assert 'gui_dialogs' not in str(list(sys.modules.keys()))\n"
        )
        env = dict(os.environ, THERMALSIM_HEADLESS="1")
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"

    def test_json_clean_handles_numpy_and_non_finite(self):
        cleaned = headless_api._json_clean({
            "f": np.float64(1.25),
            "i": np.int64(7),
            "b": np.bool_(True),
            "arr": np.array([1.0, np.nan, np.inf, -np.inf]),
        })
        assert type(cleaned["f"]) is float
        assert type(cleaned["i"]) is int
        assert type(cleaned["b"]) is bool
        assert cleaned["arr"] == [1.0, None, None, None]
        dumped = json.dumps(cleaned, allow_nan=False)
        assert json.loads(dumped) == cleaned


class TestInspectBoard:
    """Tests for inspect_board."""

    def test_inspect_mock_board_and_thermal_candidates(self, sample_2layer_board_file):
        # 8-pad IC whose pad "8" is 2.5 mm^2 (largest on >=8-pad IC, >= 2.0 mm^2)
        ic_pads = [
            MockPad(
                position=VECTOR2I(20_000_000 + i * 1_000_000, 20_000_000),
                layer=F_Cu,
                bbox=EDA_RECT(19_800_000 + i * 1_000_000, 19_800_000, 400_000, 400_000),
                net_code=10 + i,
                net_name=f"SIG{i}",
                number=str(i + 1),
            )
            for i in range(7)
        ]
        ic_pads.append(
            MockPad(
                position=VECTOR2I(23_000_000, 23_000_000),
                layer=F_Cu,
                bbox=EDA_RECT(22_000_000, 22_000_000, 2_000_000, 1_250_000),  # 2.5 mm^2
                net_code=2,
                net_name="GND",
                number="8",
            )
        )
        # Large pad >= 4.0 mm^2 on a 2-pad component
        big_pad = MockPad(
            position=VECTOR2I(40_000_000, 40_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(39_000_000, 39_000_000, 2_500_000, 2_000_000),  # 5.0 mm^2
            net_code=1,
            net_name="VIN",
            number="1",
        )
        extra_fps = [
            MockFootprint(reference="U2", pads=ic_pads),
            MockFootprint(reference="D1", pads=[big_pad]),
        ]
        board = _make_board(sample_2layer_board_file, extra_fps=extra_fps)

        info = headless_api.inspect_board(board)
        json.dumps(info, allow_nan=False)

        assert info["schema_version"] == "1.0"
        assert info["command"] == "inspect"
        assert info["board"]["width_mm"] == pytest.approx(50.0)
        assert info["board"]["height_mm"] == pytest.approx(50.0)
        assert info["stackup"]["layer_count"] == 2
        assert info["stackup"]["layers"] == ["F.Cu", "B.Cu"]
        assert info["stackup"]["total_thickness_mm"] == pytest.approx(1.6)
        assert info["zones"] == {"total": 1, "unfilled": 0}
        assert info["thermal_pad_eco1"] == {"present": True, "shape_count": 1}

        fp_refs = [fp["ref"] for fp in info["footprints"]]
        assert fp_refs == sorted(fp_refs)
        cand_names = {c["name"] for c in info["thermal_candidates"]}
        assert "U1.EP" in cand_names
        assert "U2.8" in cand_names
        assert "D1.1" in cand_names
        assert "C1.1" not in cand_names

        assert info["nets"][0]["pad_count"] >= info["nets"][-1]["pad_count"]
        assert all(n["name"] != "" for n in info["nets"])
        assert set(info["capabilities"].keys()) == {
            "has_libs", "native_core", "native_mg", "native_compact", "core_version"
        }

    def test_inspect_board_from_path_calls_loadboard(self, tmp_path, monkeypatch):
        pcb_path = tmp_path / "board.kicad_pcb"
        pcb_path.write_text("(kicad_pcb)", encoding="utf-8")
        board = _make_board(str(pcb_path))
        monkeypatch.setattr(headless_api.pcbnew, "LoadBoard", lambda p: board, raising=False)

        info = headless_api.inspect_board(pcb_path)
        assert info["board"]["name"] == "board.kicad_pcb"
        assert info["command"] == "inspect"


class TestNormalizeJobConfig:
    """Tests for normalize_job_config."""

    def test_does_not_mutate_input_and_normalizes_pad_formats(self):
        raw = {
            "power_pads": [
                {"ref": "U1", "pad": "EP", "power_w": 5.0},
                {"name": "U2.8", "power": "2.5W"},
                {"name": "D1.1", "pwl": "/tmp/profile.pwl"},
            ]
        }
        snapshot = json.dumps(raw)
        norm = headless_api.normalize_job_config(raw, suggested_res_mm=0.4)
        assert json.dumps(raw) == snapshot
        assert norm["res"] == pytest.approx(0.4)
        assert norm["steady_state"] is False  # PWL present
        assert norm["power_pads"] == [
            {"ref": "U1", "pad": "EP", "name": "U1.EP", "power": "5.0"},
            {"ref": "U2", "pad": "8", "name": "U2.8", "power": "2.5"},
            {"ref": "D1", "pad": "1", "name": "D1.1", "power": "/tmp/profile.pwl"},
        ]

    def test_dict_power_pads_and_steady_state_defaults(self):
        norm_ss = headless_api.normalize_job_config({"power_pads": {"U1.EP": 3.0}})
        assert norm_ss["steady_state"] is True
        assert norm_ss["power_pads"] == [{"ref": "U1", "pad": "EP", "name": "U1.EP", "power": "3.0"}]

        norm_tr = headless_api.normalize_job_config({"power_pads": {"U1.EP": 3.0}, "time": 15.0})
        assert norm_tr["steady_state"] is False
        assert norm_tr["time"] == pytest.approx(15.0)

        norm_explicit = headless_api.normalize_job_config(
            {"power_pads": {"U1.EP": 3.0}, "time": 15.0, "steady_state": True}
        )
        assert norm_explicit["steady_state"] is True

    def test_heatsink_inference_and_material_cp(self):
        norm = headless_api.normalize_job_config({
            "power_pads": {"U1.EP": 2.0},
            "heatsink_rth_kw": 3.5,
            "heatsink_material": "copper",
        })
        assert norm["use_heatsink"] is True
        assert norm["pad_sink_mode"] == "heatsink"
        assert norm["heatsink_rth_kw"] == pytest.approx(3.5)
        assert norm["heatsink_cp_j_kgk"] == pytest.approx(385.0)

        norm_fixed_temp = headless_api.normalize_job_config({
            "power_pads": {"U1.EP": 2.0},
            "use_heatsink": True,
            "pad_sink_temp_c": 40.0,
        })
        assert norm_fixed_temp["pad_sink_mode"] == "fixed"
        assert norm_fixed_temp["pad_sink_temp_use_ambient"] is False
        assert norm_fixed_temp["pad_sink_temp_c"] == pytest.approx(40.0)


class TestValidateJob:
    """Tests for validate_job."""

    def test_valid_job_and_case_insensitive_pad_match(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file)
        res = headless_api.validate_job(board, {"power_pads": {"u1.ep": 4.0}})
        json.dumps(res, allow_nan=False)
        assert res["valid"] is True
        assert res["ready"] is True
        assert res["errors"] == []
        assert res["resolved_power_pads"][0]["name"] == "U1.EP"
        assert res["grid"]["nodes"] > 0

    def test_missing_pad_and_missing_footprint_hints(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file)
        res = headless_api.validate_job(board, {
            "power_pads": {"U1.99": 2.0, "Z9.1": 1.0},
        })
        assert res["valid"] is False
        codes = {e["code"] for e in res["errors"]}
        assert "PAD_NOT_FOUND" in codes
        u1_err = next(e for e in res["errors"] if e["target"] == "U1.99")
        z9_err = next(e for e in res["errors"] if e["target"] == "Z9.1")
        assert "Available pads on U1:" in u1_err["hint"]
        assert "EP" in u1_err["hint"]
        assert "Available footprints:" in z9_err["hint"]

    def test_unfilled_zones_error_vs_refill_warning(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file, unfilled_zone=True)
        err_res = headless_api.validate_job(board, {"power_pads": {"U1.EP": 2.0}})
        assert err_res["valid"] is False
        assert any(e["code"] == "UNFILLED_ZONES" for e in err_res["errors"])

        warn_res = headless_api.validate_job(
            board, {"power_pads": {"U1.EP": 2.0}, "refill_zones": True}
        )
        assert warn_res["valid"] is True
        assert any(w["code"] == "UNFILLED_ZONES_WILL_REFILL" for w in warn_res["warnings"])

    def test_no_heat_source_and_no_eco1_warning(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file, with_eco1=False)
        res = headless_api.validate_job(board, {"heatsink_rth_kw": 4.0})
        assert res["valid"] is False
        assert any(e["code"] == "NO_HEAT_SOURCE" for e in res["errors"])
        assert any(w["code"] == "NO_ECO1_SHAPES" for w in res["warnings"])

    def test_pwl_validation(self, sample_2layer_board_file, tmp_path):
        board = _make_board(sample_2layer_board_file)
        pwl_file = tmp_path / "pulse.pwl"
        pwl_file.write_text("0.0 0.0\n1.0 5.0\n2.0 5.0\n", encoding="utf-8")

        ok = headless_api.validate_job(board, {"power_pads": {"U1.EP": str(pwl_file)}})
        assert ok["valid"] is True
        assert ok["normalized_config"]["steady_state"] is False

        bad = headless_api.validate_job(board, {"power_pads": {"U1.EP": str(tmp_path / "missing.pwl")}})
        assert bad["valid"] is False
        assert any(e["code"] == "INVALID_PWL" for e in bad["errors"])

    def test_current_groups_balance_validation(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file)
        unbalanced = headless_api.validate_job(board, {
            "current_enabled": True,
            "current_groups": [
                {"name": "VIN", "pads": {"U1.1": 3.0, "C1.1": -1.0}},
            ],
        })
        assert unbalanced["valid"] is False
        assert any(e["code"] == "CURRENT_UNBALANCED" for e in unbalanced["errors"])

        balanced = headless_api.validate_job(board, {
            "current_enabled": True,
            "current_groups": [
                {"name": "VIN", "pads": {"U1.1": 3.0, "C1.1": -3.0}},
            ],
        })
        assert balanced["valid"] is True


class TestBuildSemanticSummary:
    """Tests for build_semantic_summary."""

    def test_hotspots_suppression_and_limits_evaluation(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file)
        rows, cols = 100, 100
        res = 0.5
        T = np.full((2, rows, cols), 25.0, dtype=np.float64)
        # Peak 1 near U1 (10mm, 10mm -> r=20, c=20) with broad plateau
        T[0, 18:23, 18:23] = 120.0
        T[0, 20, 20] = 125.0
        # Peak 2 near C1 (35mm, 35mm -> r=70, c=70)
        T[0, 70, 70] = 110.0

        dummy_result = SolverResult(
            T=T,
            aborted=False,
            step_counter=1,
            total_solve_time=0.02,
            total_factor_time=0.01,
            factor_count=1,
            snapshot_stats=[],
            snapshot_files=[],
            phase_metrics=[],
            balance_history=[],
            k_norm_info={
                "backend": "SciPy",
                "strategy": "direct_steady_state",
                "steady_state": True,
                "N": T.size,
                "pin_w": 5.0,
                "pout_final_w": 5.0,
                "steady_rel_diff": 0.0,
                "heatsink_enabled": True,
                "heatsink_mode": "heatsink",
                "heatsink_temp_c": 62.0,
                "heatsink_temp_rise_c": 37.0,
                "heatsink_rth_kw": 5.0,
                "heatsink_capacity_j_k": 44.85,
                "heatsink_tau_s": 224.25,
                "heatsink_power_w": 7.4,
                "pcb_convection_power_w": 1.0,
                "heatsink_power_fraction": 0.88,
                "heatsink_pad_flux_w": 7.4,
            },
        )
        settings = headless_api.normalize_job_config({
            "power_pads": {"U1.EP": 5.0},
            "limits": {
                "max_board_c": 120.0,
                "max_heatsink_c": 60.0,
                "components": {"c*": 100.0, "U1": 130.0},
            },
        })
        summary = headless_api.build_semantic_summary(
            T=T,
            board=board,
            layer_names=["F.Cu", "B.Cu"],
            x_min=0.0,
            y_min=0.0,
            res=res,
            settings=settings,
            result=dummy_result,
            elapsed_s=0.05,
        )
        json.dumps(summary, allow_nan=False)

        assert summary["status"] == "limit_violation"
        assert summary["temperatures"]["max_c"] == pytest.approx(125.0)
        assert summary["temperatures"]["max_rise_k"] == pytest.approx(100.0)
        assert len(summary["hotspots"]) == 2
        assert summary["hotspots"][0]["nearest_component"]["ref"] == "U1"
        assert summary["hotspots"][1]["nearest_component"]["ref"] == "C1"
        dist_cells = np.hypot(
            (summary["hotspots"][0]["x_mm"] - summary["hotspots"][1]["x_mm"]) / res,
            (summary["hotspots"][0]["y_mm"] - summary["hotspots"][1]["y_mm"]) / res,
        )
        assert dist_cells >= max(3, int(round(3.0 / res)))

        assert list(summary["components"].keys()) == ["U1", "C1"]
        assert summary["top_components"][0]["ref"] == "U1"

        lim = summary["limits_check"]
        assert lim["enabled"] is True
        assert lim["passed"] is False
        rules = {(v["rule"], v["target"]): v for v in lim["violations"]}
        assert ("max_board_c", "board") in rules
        assert ("max_heatsink_c", "heatsink") in rules
        assert ("component_max_c", "C1") in rules
        assert rules[("max_board_c", "board")]["margin_c"] == pytest.approx(-5.0)


class TestRunAndSweepAndCompare:
    """End-to-end headless solve, sweep, and compare tests."""

    def test_run_validation_error_returns_structured_payload(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file)
        res = headless_api.run_headless_simulation(board, {"power_pads": {"U1.999": 2.0}})
        assert res["status"] == "validation_error"
        assert res["valid"] is False
        assert res["errors"][0]["code"] == "PAD_NOT_FOUND"

    def test_run_steady_state_and_scales_and_artifacts(self, sample_2layer_board_file, tmp_path):
        board = _make_board(sample_2layer_board_file)
        out1 = tmp_path / "run1"
        res1 = headless_api.run_headless_simulation(
            board,
            {
                "power_pads": {"U1.EP": 2.0},
                "res": 2.0,
                "compute_engine": "legacy",
                "limits": {"max_board_c": 200.0},
            },
            out_dir=out1,
            generate_plots=True,
            generate_html=True,
        )
        json.dumps(res1, allow_nan=False)
        assert res1["status"] == "ok"
        assert res1["temperatures"]["max_c"] > 25.0
        assert os.path.isfile(res1["artifacts"]["summary_json"])
        assert os.path.isfile(res1["artifacts"]["heatmap_png"])
        assert os.path.isfile(res1["artifacts"]["preview_png"])
        assert os.path.isfile(res1["artifacts"]["report_html"])

        res2 = headless_api.run_headless_simulation(
            board,
            {
                "power_pads": {"U1.EP": 2.0},
                "res": 2.0,
                "compute_engine": "legacy",
                "power_scale": 2.0,
            },
            generate_plots=False,
            generate_html=False,
        )
        assert res2["temperatures"]["max_rise_k"] == pytest.approx(
            2.0 * res1["temperatures"]["max_rise_k"], rel=1e-4
        )

        res_thick_cu = headless_api.run_headless_simulation(
            board,
            {
                "power_pads": {"U1.EP": 2.0},
                "res": 2.0,
                "compute_engine": "legacy",
                "copper_thickness_scale": 3.0,
            },
            generate_plots=False,
            generate_html=False,
        )
        assert res_thick_cu["temperatures"]["max_c"] < res1["temperatures"]["max_c"]

    def test_run_dynamic_heatsink_transient_fast_engine(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file, with_eco1=True)
        res = headless_api.run_headless_simulation(
            board,
            {
                "power_pads": {"U1.EP": 5.0},
                "time": 10.0,
                "res": 2.0,
                "compute_engine": "fast_native",
                "mesh_mode": "uniform",
                "heatsink_rth_kw": 4.0,
                "heatsink_mass_g": 25.0,
            },
            generate_plots=False,
            generate_html=False,
        )
        assert res["status"] == "ok"
        assert res["heatsink"]["enabled"] is True
        assert res["heatsink"]["mode"] == "heatsink"
        assert res["heatsink"]["temp_c"] > 25.0
        assert res["heatsink"]["tau_s"] > 0.0

    def test_run_adaptive_mesh_fast_engine(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file, with_eco1=True)
        res = headless_api.run_headless_simulation(
            board,
            {
                "power_pads": {"U1.EP": 3.0},
                "res": 1.0,
                "compute_engine": "fast_native",
                "mesh_mode": "adaptive",
                "adaptive_max_cell_ratio": 4,
            },
            generate_plots=False,
            generate_html=False,
        )
        assert res["status"] == "ok"
        assert res["solver"]["mesh_mode"] == "adaptive"
        assert res["solver"]["adaptive_nodes"] <= res["solver"]["nodes"]
        assert res["temperatures"]["max_c"] > 25.0

    def test_run_refills_zones_when_requested(self, sample_2layer_board_file, monkeypatch):
        board = _make_board(sample_2layer_board_file, unfilled_zone=True)
        fill_calls = []

        class RecordingFiller:
            def __init__(self, b):
                self.b = b

            def Fill(self, zones):
                fill_calls.append(len(list(zones)))
                for z in zones:
                    z._filled = True

        monkeypatch.setattr(headless_api.pcbnew, "ZONE_FILLER", RecordingFiller, raising=False)
        res = headless_api.run_headless_simulation(
            board,
            {"power_pads": {"U1.EP": 2.0}, "res": 2.0},
            generate_plots=False,
            generate_html=False,
            refill_zones=True,
        )
        assert res["status"] == "ok"
        assert fill_calls == [1]

    def test_run_electrical_joule_heating(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file)
        res = headless_api.run_headless_simulation(
            board,
            {
                "res": 2.0,
                "compute_engine": "legacy",
                "current_enabled": True,
                "current_groups": [{"name": "VIN", "pads": {"U1.1": 5.0, "C1.1": -5.0}}],
            },
            generate_plots=False,
            generate_html=False,
        )
        assert res["status"] == "ok"
        assert res["electrical"] is not None
        assert res["electrical"]["total_loss_w"] > 0.0
        assert res["temperatures"]["max_c"] > 25.0

    def test_parameter_sweep_reuses_geometry_state_and_compares(
        self, sample_2layer_board_file, tmp_path, monkeypatch
    ):
        board = _make_board(sample_2layer_board_file, with_eco1=True)
        geom_calls = []
        orig_build_geom = headless_api.build_geometry_state

        def spy_geom(*args, **kwargs):
            geom_calls.append(1)
            return orig_build_geom(*args, **kwargs)

        monkeypatch.setattr(headless_api, "build_geometry_state", spy_geom)

        sweep_dir = tmp_path / "sweep"
        sweep = headless_api.run_parameter_sweep(
            board,
            {
                "power_pads": {"U1.EP": 4.0},
                "res": 2.0,
                "compute_engine": "legacy",
                "heatsink_rth_kw": 5.0,
                "limits": {"max_board_c": 45.0},
            },
            param="heatsink_rth_kw",
            values=[2.0, 5.0, 15.0],
            out_dir=sweep_dir,
        )
        json.dumps(sweep, allow_nan=False)
        assert len(geom_calls) == 1
        assert sweep["status"] == "ok"
        assert sweep["best_value"] == pytest.approx(2.0)
        assert sweep["runs"][0]["max_temp_c"] < sweep["runs"][2]["max_temp_c"]
        assert os.path.isfile(sweep_dir / "sweep_summary.json")
        assert os.path.isfile(sweep["runs"][0]["summary_json"])

        cmp_dict = headless_api.compare_runs(
            sweep["runs"][0]["summary_json"],
            os.path.dirname(sweep["runs"][2]["summary_json"]),
        )
        json.dumps(cmp_dict, allow_nan=False)
        assert cmp_dict["command"] == "compare"
        assert cmp_dict["delta_max_temp_c"] > 0.0
        assert cmp_dict["delta_heatsink_temp_c"] > 0.0
        assert len(cmp_dict["layer_deltas"]) == 2
        assert len(cmp_dict["component_deltas"]) >= 1

    def test_compare_runs_limits_resolved_and_new_violations(self):
        run_a = {
            "temperatures": {"max_c": 110.0, "min_c": 30.0, "mean_c": 50.0, "layers": []},
            "heatsink": {"temp_c": 65.0},
            "components": {"U1": {"max_c": 110.0}},
            "limits_check": {
                "enabled": True,
                "passed": False,
                "violations": [
                    {"rule": "max_board_c", "target": "board", "limit_c": 100.0, "actual_c": 110.0, "margin_c": -10.0}
                ],
            },
        }
        run_b = {
            "temperatures": {"max_c": 95.0, "min_c": 28.0, "mean_c": 45.0, "layers": []},
            "heatsink": {"temp_c": 70.0},
            "components": {"U1": {"max_c": 95.0}},
            "limits_check": {
                "enabled": True,
                "passed": False,
                "violations": [
                    {"rule": "max_heatsink_c", "target": "heatsink", "limit_c": 65.0, "actual_c": 70.0, "margin_c": -5.0}
                ],
            },
        }
        cmp_res = headless_api.compare_runs(run_a, run_b)
        assert cmp_res["delta_max_temp_c"] == pytest.approx(-15.0)
        assert cmp_res["delta_heatsink_temp_c"] == pytest.approx(5.0)
        assert len(cmp_res["limits_comparison"]["resolved_violations"]) == 1
        assert cmp_res["limits_comparison"]["resolved_violations"][0]["rule"] == "max_board_c"
        assert len(cmp_res["limits_comparison"]["new_violations"]) == 1
        assert cmp_res["limits_comparison"]["new_violations"][0]["rule"] == "max_heatsink_c"

    def test_parameter_sweep_rejects_invalid_param(self, sample_2layer_board_file):
        board = _make_board(sample_2layer_board_file)
        res = headless_api.run_parameter_sweep(
            board,
            {"power_pads": {"U1.EP": 2.0}},
            param="unknown_param",
            values=[1.0, 2.0],
        )
        assert res["status"] == "validation_error"
        assert res["errors"][0]["code"] == "INVALID_SWEEP_PARAM"
