"""Tests for project-scoped heat-source and current-path configuration."""

import json
import math
import os

import pytest

from ThermalSim.source_configuration import (
    SCHEMA_VERSION,
    CurrentCircuitDefinition,
    CurrentPathDefinition,
    HeatSourceDefinition,
    PadRef,
    ProjectConfiguration,
    expand_current_paths,
    expand_heat_sources,
    load_project_config,
    migrate_v2_settings,
    save_project_config,
)


def _pad(number, area=None, net="VIN", uuid_value=""):
    """Build a concise pad fixture with stable display data."""
    return PadRef(
        pad_uuid=uuid_value,
        footprint_uuid="fp-1",
        reference="U1",
        pad_number=str(number),
        display_name="U1-%s [%s]" % (number, net),
        net_name=net,
        net_code=7,
        area_mm2=area,
    )


def test_sidecar_round_trip_is_schema_v3_and_resolves_pwl_path(tmp_path):
    """Saving uses sidecar-relative PWL paths and loading makes them absolute."""
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profile_path = profile_dir / "regulator.pwl"
    profile_path.write_text("0 0\n1 3\n", encoding="utf-8")
    sidecar = tmp_path / "example.kicad_pcb.thermalsim.json"
    config = ProjectConfiguration(
        heat_sources=[
            HeatSourceDefinition(
                id="regulator",
                name="Regulator loss",
                profile_kind="pwl",
                pwl_path=str(profile_path),
                pads=[_pad("1")],
                distribution="equal",
            )
        ],
        current_paths=[
            CurrentPathDefinition(
                id="vin",
                name="VIN",
                net_name="VIN",
                current_a=2.0,
                source_pads=[_pad("1")],
                sink_pads=[_pad("2")],
            )
        ],
    )

    result = save_project_config(sidecar, config)

    assert result == sidecar
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    stored_path = raw["heat_sources"][0]["power_profile"]["path"]
    assert not os.path.isabs(stored_path)
    assert os.path.normpath(os.path.join(tmp_path, stored_path)) == str(profile_path)
    assert not list(tmp_path.glob("*.tmp"))

    loaded = load_project_config(sidecar)

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.heat_sources[0].pwl_path == str(profile_path)
    assert loaded.current_paths[0].source_shares == [1.0]


def test_heat_source_expansion_preserves_total_and_falls_back_to_equal():
    """Area, custom, and invalid-area source allocations always sum to one."""
    first = _pad("1", area=2.0)
    second = _pad("2", area=6.0)
    by_area = HeatSourceDefinition(
        id="area",
        name="Area source",
        power_w=8.0,
        distribution="area",
        pads=[first, second],
    )
    expanded = expand_heat_sources([by_area])

    assert [item["share"] for item in expanded] == pytest.approx([0.25, 0.75])
    assert [item["profile"]["value_w"] for item in expanded] == pytest.approx([2.0, 6.0])
    assert math.fsum(item["profile"]["value_w"] for item in expanded) == pytest.approx(8.0)
    assert all(item["distribution_used"] == "area" for item in expanded)

    fallback = HeatSourceDefinition(
        id="fallback",
        name="Fallback source",
        power_w=2.0,
        distribution="area",
        pads=[_pad("3", area=1.0), _pad("4", area=None)],
    )
    custom = HeatSourceDefinition(
        id="custom",
        name="Custom source",
        profile_kind="pwl",
        pwl_path="load.pwl",
        distribution="custom",
        pads=[first, second],
        custom_shares=[2.0, 1.0],
    )
    fallback_expanded = expand_heat_sources([fallback])
    custom_expanded = expand_heat_sources([custom])

    assert [item["share"] for item in fallback_expanded] == pytest.approx([0.5, 0.5])
    assert all(item["distribution_used"] == "equal" for item in fallback_expanded)
    assert [item["profile"]["scale"] for item in custom_expanded] == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert math.fsum(item["profile"]["scale"] for item in custom_expanded) == pytest.approx(1.0)


def test_current_path_expansion_uses_circuit_current_and_is_balanced():
    """Circuit-linked paths emit positive sources and negative sinks at one magnitude."""
    path = CurrentPathDefinition(
        id="supply",
        name="Supply net",
        net_name="VIN",
        current_a=5.0,
        circuit_id="load-loop",
        source_pads=[_pad("1"), _pad("2")],
        sink_pads=[_pad("3"), _pad("4")],
        source_shares=[2.0, 1.0],
        sink_shares=[1.0, 3.0],
    )
    circuit = CurrentCircuitDefinition(
        id="load-loop", name="Load loop", current_a=6.0, path_ids=["supply"]
    )

    terminals = expand_current_paths([path], [circuit])

    assert [terminal["role"] for terminal in terminals] == ["source", "source", "sink", "sink"]
    assert [terminal["current_a"] for terminal in terminals[:2]] == pytest.approx([4.0, 2.0])
    assert [terminal["current_a"] for terminal in terminals[2:]] == pytest.approx([-1.5, -4.5])
    assert math.fsum(item["current_a"] for item in terminals) == pytest.approx(0.0, abs=1e-15)
    assert all(item["net_name"] == "VIN" for item in terminals)
    assert expand_current_paths(
        ProjectConfiguration(current_paths=[path], current_circuits=[circuit])
    ) == terminals


def test_v2_migration_preserves_power_and_marks_unbalanced_nets_disabled():
    """Legacy pad values become sources, while unsafe legacy current nets cannot run."""
    v2 = {
        "schema_version": 2,
        "power_str": "9.0",
        "power_pads": [
            {
                "pad_key": "U1:1:7:100:200",
                "name": "U1-1 [VIN]",
                "net_name": "VIN",
                "net_code": 7,
                "power": "2.5",
            },
            {
                "pad_key": "U2:1:8:300:400",
                "name": "U2-1 [VOUT]",
                "net_name": "VOUT",
                "net_code": 8,
                "power": "profiles/load.pwl",
            },
        ],
        "current_enabled": True,
        "current_groups": [
            {
                "name": "Input",
                "mode": "per_pad",
                "pads": [{"pad_key": "J1:1:7:1:1", "name": "J1-1 [VIN]", "net_name": "VIN", "current_a": 6.0}],
            },
            {
                "name": "Load",
                "mode": "per_pad",
                "pads": [
                    {"pad_key": "U3:1:7:2:2", "name": "U3-1 [VIN]", "net_name": "VIN", "current_a": -4.0},
                    {"pad_key": "U3:2:7:3:3", "name": "U3-2 [VIN]", "net_name": "VIN", "current_a": -2.0},
                    {"pad_key": "J2:1:9:4:4", "name": "J2-1 [BAD]", "net_name": "BAD", "current_a": 3.0},
                ],
            },
        ],
        "res": 0.25,
    }

    config = migrate_v2_settings(v2)

    assert config.schema_version == SCHEMA_VERSION
    assert [source.profile_kind for source in config.heat_sources] == ["constant", "pwl"]
    assert config.heat_sources[0].power_w == pytest.approx(2.5)
    assert config.heat_sources[1].pwl_path == "profiles/load.pwl"
    assert config.heat_sources[0].pads[0].reference == "U1"
    assert config.heat_sources[0].pads[0].pad_number == "1"
    assert config.extra["res"] == 0.25

    by_net = {path.net_name: path for path in config.current_paths}
    vin = by_net["VIN"]
    assert vin.enabled is True
    assert vin.needs_repair is False
    assert vin.current_a == pytest.approx(6.0)
    assert vin.source_shares == pytest.approx([1.0])
    assert vin.sink_shares == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    bad = by_net["BAD"]
    assert bad.enabled is False
    assert bad.needs_repair is True
    assert "source and sink" in bad.repair_reason
    terminals = expand_current_paths(config.current_paths)
    assert math.fsum(item["current_a"] for item in terminals) == pytest.approx(0.0)
    assert {item["net_name"] for item in terminals} == {"VIN"}


def test_enabled_broken_current_path_and_missing_circuit_do_not_expand():
    """Expansion fails closed instead of manufacturing terminal balance for bad input."""
    incomplete = CurrentPathDefinition(
        id="incomplete",
        name="Incomplete",
        net_name="VIN",
        current_a=1.0,
        source_pads=[_pad("1")],
    )
    with pytest.raises(ValueError, match="source and one sink"):
        expand_current_paths([incomplete])

    missing_circuit = CurrentPathDefinition(
        id="missing-circuit",
        name="Missing circuit",
        net_name="VIN",
        current_a=1.0,
        circuit_id="unknown",
        source_pads=[_pad("1")],
        sink_pads=[_pad("2")],
    )
    with pytest.raises(ValueError, match="missing circuit"):
        expand_current_paths([missing_circuit])


def test_enabled_broken_heat_source_does_not_expand():
    """Repair-marked or non-zero padless sources must fail closed."""
    broken = HeatSourceDefinition(
        id="broken", name="Broken source", power_w=1.0,
        pads=[_pad("1")], needs_repair=True,
    )
    padless = HeatSourceDefinition(
        id="padless", name="Padless source", power_w=1.0,
    )

    with pytest.raises(ValueError, match="needs repair"):
        expand_heat_sources([broken])
    with pytest.raises(ValueError, match="at least one pad"):
        expand_heat_sources([padless])
