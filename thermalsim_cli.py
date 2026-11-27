"""Headless CLI entrypoint for ThermalSim.

Supports both package execution (``python3 -m KiCad_Thermal_Sim <cmd>``)
and standalone script execution (``python3 thermalsim_cli.py <cmd>``).
Stdout is strictly reserved for a single JSON document per invocation; any
solver or library output on stdout is redirected to stderr.
"""

import os

if __name__ == "__main__":
    os.environ.setdefault("THERMALSIM_HEADLESS", "1")

import argparse
import contextlib
import importlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = "1.0"

# Module-level hook for test injection; when None, _get_headless_api() loads it.
headless_api = None


class _CliUsageError(Exception):
    """Raised on invalid CLI arguments, flag syntax, or config file errors."""


class _JsonArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises _CliUsageError instead of writing to stderr and exiting."""

    def error(self, message: str) -> None:
        raise _CliUsageError(message)


def _get_headless_api():
    """Import and return headless_api cleanly for both package and script invocation."""
    if headless_api is not None:
        return headless_api
    if __package__:
        from . import headless_api as api_mod
        return api_mod

    pkg_dir = Path(__file__).resolve().parent
    parent_dir = pkg_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    pkg_name = pkg_dir.name
    try:
        return importlib.import_module(f"{pkg_name}.headless_api")
    except ImportError:
        if str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))
        return importlib.import_module("headless_api")


def _finite_float(raw: str) -> float:
    """Parse a finite float for argparse arguments."""
    try:
        val = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {raw!r}") from exc
    if not math.isfinite(val):
        raise argparse.ArgumentTypeError(f"non-finite float value: {raw!r}")
    return val


def _parse_pad_spec(spec: str) -> Dict[str, str]:
    """Parse ``REF.PAD=VALUE`` (stripping optional trailing ``W``/``w`` on floats)."""
    if "=" not in spec:
        raise _CliUsageError(f"Invalid --pad syntax {spec!r}; expected REF.PAD=VALUE.")
    target, raw_val = spec.split("=", 1)
    target = target.strip()
    raw_val = raw_val.strip()
    if "." not in target:
        raise _CliUsageError(f"Invalid --pad target {target!r}; expected REF.PAD.")
    ref, pad = target.split(".", 1)
    ref = ref.strip()
    pad = pad.strip()
    if not ref or not pad or not raw_val:
        raise _CliUsageError(f"Invalid --pad syntax {spec!r}; ref, pad, and value must be non-empty.")

    power_str: Optional[str] = None
    if raw_val.lower().endswith("w") and not os.path.exists(raw_val):
        num_part = raw_val[:-1].strip()
        if not num_part:
            raise _CliUsageError(f"Invalid --pad power value {raw_val!r} in {spec!r}.")
        try:
            fval = float(num_part)
        except ValueError as exc:
            raise _CliUsageError(f"Invalid --pad power value {raw_val!r} in {spec!r}.") from exc
        if not math.isfinite(fval):
            raise _CliUsageError(f"Non-finite --pad power value {raw_val!r} in {spec!r}.")
        power_str = str(fval)
    else:
        try:
            fval = float(raw_val)
        except ValueError:
            # Treat non-numeric value as a PWL CSV path; validate_job checks the file.
            power_str = raw_val
        else:
            if not math.isfinite(fval):
                raise _CliUsageError(f"Non-finite --pad power value {raw_val!r} in {spec!r}.")
            power_str = str(fval)

    return {
        "ref": ref,
        "pad": pad,
        "name": f"{ref}.{pad}",
        "power": power_str,
    }


def _parse_flow_side(raw_side: str, side_name: str) -> List[Tuple[str, Optional[float]]]:
    """Parse one side of a ``--current-flow`` spec into ``[(pad_name, current_a_or_None)]``."""
    tokens = [tok.strip() for tok in raw_side.split(",") if tok.strip()]
    if not tokens:
        raise _CliUsageError(f"Empty {side_name} pad list in --current-flow specification.")
    parsed: List[Tuple[str, Optional[float]]] = []
    for tok in tokens:
        if ":" in tok:
            pad_part, val_part = tok.rsplit(":", 1)
            pad_part = pad_part.strip()
            val_part = val_part.strip()
            if val_part.lower().endswith("a"):
                val_part = val_part[:-1].strip()
            if not pad_part or "." not in pad_part or not val_part:
                raise _CliUsageError(
                    f"Invalid pad/current token {tok!r} in --current-flow; expected REF.PAD or REF.PAD:CURRENT[A]."
                )
            try:
                fval = float(val_part)
            except ValueError as exc:
                raise _CliUsageError(f"Invalid current value {val_part!r} in --current-flow.") from exc
            if not math.isfinite(fval) or fval < 0.0:
                raise _CliUsageError(f"Current value {val_part!r} in --current-flow must be >= 0 and finite.")
            parsed.append((pad_part, fval))
        else:
            if "." not in tok:
                raise _CliUsageError(f"Invalid pad identifier {tok!r} in --current-flow; expected REF.PAD.")
            ref_p, num_p = tok.split(".", 1)
            if not ref_p.strip() or not num_p.strip():
                raise _CliUsageError(f"Invalid pad identifier {tok!r} in --current-flow; expected REF.PAD.")
            parsed.append((tok, None))
    return parsed


def _parse_current_flow_spec(spec: str) -> Dict[str, Any]:
    """Parse ``FROM->TO`` current flow CLI spec (e.g. ``'J1.1->U1.1:3,U2.1:3,U3.1:3'`` or ``'J1.1->U1.1,U1.2:12A'``)."""
    if "->" not in spec:
        raise _CliUsageError(
            f"Invalid --current-flow syntax {spec!r}; expected 'FROM->TO' (e.g. 'J1.1->U1.1:3,U2.1:3')."
        )
    lhs, rhs = spec.split("->", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs or not rhs:
        raise _CliUsageError(f"Invalid --current-flow syntax {spec!r}; both source and sink sides are required.")

    explicit_total: Optional[float] = None
    if "=" in rhs:
        rhs_pads, total_raw = rhs.rsplit("=", 1)
        rhs = rhs_pads.strip()
        total_clean = total_raw.strip()
        if total_clean.lower().endswith("a"):
            total_clean = total_clean[:-1].strip()
        try:
            explicit_total = float(total_clean)
        except ValueError as exc:
            raise _CliUsageError(f"Invalid total current {total_raw!r} in --current-flow.") from exc
        if not math.isfinite(explicit_total) or explicit_total < 0.0:
            raise _CliUsageError(f"Total current {total_raw!r} in --current-flow must be >= 0 and finite.")

    from_items = _parse_flow_side(lhs, "source")
    to_items = _parse_flow_side(rhs, "sink")

    def _finalize_side(
        items: List[Tuple[str, Optional[float]]],
        side_label: str,
    ) -> Tuple[Any, str, Optional[float]]:
        vals = [v for _, v in items]
        if all(v is not None for v in vals):
            return {p: float(v) for p, v in items if v is not None}, "fixed", None
        if all(v is None for v in vals):
            return [p for p, _ in items], "equipotential", None
        if len(items) >= 2 and all(v is None for v in vals[:-1]) and vals[-1] is not None:
            return [p for p, _ in items], "equipotential", float(vals[-1])
        raise _CliUsageError(
            f"Ambiguous {side_label} current specification in --current-flow {spec!r}; "
            "specify current on every pad (REF.PAD:A) or once at the end for equipotential sharing."
        )

    from_sel, from_sharing, from_tail_total = _finalize_side(from_items, "source")
    to_sel, to_sharing, to_tail_total = _finalize_side(to_items, "sink")

    flow: Dict[str, Any] = {
        "from": from_sel,
        "from_sharing": from_sharing,
        "to": to_sel,
        "to_sharing": to_sharing,
    }
    inferred_total = explicit_total if explicit_total is not None else (
        to_tail_total if to_tail_total is not None else from_tail_total
    )
    if inferred_total is not None:
        flow["total_current_a"] = float(inferred_total)
    elif from_sharing != "fixed" and to_sharing != "fixed":
        raise _CliUsageError(
            f"Missing current value in --current-flow {spec!r}; add ':12A' or '=12A' or per-pad ':A' values."
        )
    return flow


def _parse_limit_comp_spec(spec: str) -> Tuple[str, float]:
    """Parse ``PATTERN=MAX_TEMP_C`` for component thermal limits."""
    if "=" not in spec:
        raise _CliUsageError(f"Invalid --limit-comp syntax {spec!r}; expected PATTERN=TEMP_C.")
    pattern, raw_temp = spec.split("=", 1)
    pattern = pattern.strip()
    raw_temp = raw_temp.strip()
    if not pattern or not raw_temp:
        raise _CliUsageError(f"Invalid --limit-comp syntax {spec!r}; pattern and temperature must be non-empty.")
    try:
        temp_c = float(raw_temp)
    except ValueError as exc:
        raise _CliUsageError(f"Invalid --limit-comp temperature {raw_temp!r} in {spec!r}.") from exc
    if not math.isfinite(temp_c):
        raise _CliUsageError(f"Non-finite --limit-comp temperature {raw_temp!r} in {spec!r}.")
    return pattern, temp_c


def _parse_values_list(raw: str) -> List[float]:
    """Parse a comma-separated list of numeric sweep values."""
    if not raw or not raw.strip():
        raise _CliUsageError("Invalid --values argument: expected a comma-separated list of numbers.")
    parts = [part.strip() for part in raw.split(",")]
    if any(not part for part in parts):
        raise _CliUsageError(f"Invalid --values syntax {raw!r}: empty entry in list.")
    values: List[float] = []
    for part in parts:
        try:
            val = float(part)
        except ValueError as exc:
            raise _CliUsageError(f"Invalid --values entry {part!r}: expected a number.") from exc
        if not math.isfinite(val):
            raise _CliUsageError(f"Non-finite --values entry {part!r}.")
        values.append(val)
    return values


def _load_config_file(path_str: str) -> Dict[str, Any]:
    """Load and validate a JSON job config file."""
    path = Path(path_str)
    if not path.is_file():
        raise _CliUsageError(f"Config file not found: {path_str}")
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise _CliUsageError(f"Failed to read JSON config {path_str}: {exc}") from exc
    if not isinstance(data, dict):
        raise _CliUsageError(f"Config file {path_str} must contain a JSON object.")
    return data


def _pad_entry_key(entry: Any) -> Optional[Tuple[str, str]]:
    """Extract normalized (REF, PAD) key from a power_pads entry if possible."""
    if not isinstance(entry, dict):
        return None
    ref = entry.get("ref")
    pad = entry.get("pad")
    if ref is not None and pad is not None:
        return (str(ref).strip().upper(), str(pad).strip().upper())
    name = entry.get("name")
    if isinstance(name, str) and "." in name:
        r, p = name.split(".", 1)
        return (r.strip().upper(), p.strip().upper())
    return None


def _merge_power_pads(existing: Any, cli_pads: List[Dict[str, str]]) -> List[Any]:
    """Merge CLI ``--pad`` specs onto an existing ``power_pads`` list or dict."""
    if existing is None:
        merged: List[Any] = []
    elif isinstance(existing, dict):
        merged = [{"name": str(k), "power": v} for k, v in existing.items()]
    elif isinstance(existing, list):
        merged = [dict(item) if isinstance(item, dict) else item for item in existing]
    else:
        raise _CliUsageError("Config field 'power_pads' must be a list or object.")

    for entry in cli_pads:
        key = (entry["ref"].upper(), entry["pad"].upper())
        replaced = False
        for idx, prev in enumerate(merged):
            if _pad_entry_key(prev) == key:
                merged[idx] = dict(entry)
                replaced = True
                break
        if not replaced:
            merged.append(dict(entry))
    return merged


def _build_job_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build merged job_config dict from optional ``--config`` and CLI overrides."""
    config: Dict[str, Any] = {}
    if getattr(args, "config", None):
        config = _load_config_file(args.config)

    if getattr(args, "pad", None):
        parsed_pads = [_parse_pad_spec(spec) for spec in args.pad]
        config["power_pads"] = _merge_power_pads(config.get("power_pads"), parsed_pads)

    if getattr(args, "current_flow", None):
        existing_flows = config.get("current_flows")
        if existing_flows is None:
            flows_list: List[Any] = []
        elif isinstance(existing_flows, list):
            flows_list = list(existing_flows)
        else:
            raise _CliUsageError("Config field 'current_flows' must be a list.")
        for spec in args.current_flow:
            flows_list.append(_parse_current_flow_spec(spec))
        config["current_flows"] = flows_list
        config["current_enabled"] = True

    if getattr(args, "electro_thermal_coupling", None) is not None:
        config["electro_thermal_coupling"] = bool(args.electro_thermal_coupling)

    simple_mappings = [
        ("steady_state", "steady_state"),
        ("time", "time"),
        ("amb", "amb"),
        ("res", "res"),
        ("thick", "thick"),
        ("h_conv", "h_conv"),
        ("h_conv_bottom", "h_conv_bottom"),
        ("via_plating_um", "via_plating_um"),
        ("via_fill", "via_fill"),
        ("detail", "grid_detail_level"),
        ("node_budget", "grid_node_budget"),
        ("mesh", "mesh_mode"),
        ("engine", "compute_engine"),
        ("use_heatsink", "use_heatsink"),
        ("sink_mode", "pad_sink_mode"),
        ("heatsink_rth", "heatsink_rth_kw"),
        ("heatsink_mass", "heatsink_mass_g"),
        ("heatsink_material", "heatsink_material"),
        ("heatsink_cp", "heatsink_cp_j_kgk"),
        ("pad_k", "pad_k"),
        ("pad_th", "pad_th"),
        ("contact_factor", "pad_contact_factor"),
        ("power_scale", "power_scale"),
        ("copper_thickness_scale", "copper_thickness_scale"),
        ("refill_zones", "refill_zones"),
    ]
    for arg_name, cfg_key in simple_mappings:
        val = getattr(args, arg_name, None)
        if val is not None:
            config[cfg_key] = val

    if getattr(args, "sink_temp", None) is not None:
        config["pad_sink_temp_c"] = args.sink_temp
        config["pad_sink_temp_use_ambient"] = False

    limit_board = getattr(args, "limit_board", None)
    limit_heatsink = getattr(args, "limit_heatsink", None)
    limit_comp = getattr(args, "limit_comp", None)
    if limit_board is not None or limit_heatsink is not None or limit_comp:
        existing_limits = config.get("limits")
        if existing_limits is None:
            limits: Dict[str, Any] = {}
        elif isinstance(existing_limits, dict):
            limits = dict(existing_limits)
        else:
            raise _CliUsageError("Config field 'limits' must be an object.")

        if limit_board is not None:
            limits["max_board_c"] = limit_board
        if limit_heatsink is not None:
            limits["max_heatsink_c"] = limit_heatsink
        if limit_comp:
            existing_comp = limits.get("components")
            if existing_comp is None:
                comp_limits: Dict[str, float] = {}
            elif isinstance(existing_comp, dict):
                comp_limits = dict(existing_comp)
            else:
                raise _CliUsageError("Config field 'limits.components' must be an object.")
            for spec in limit_comp:
                pattern, temp_c = _parse_limit_comp_spec(spec)
                comp_limits[pattern] = temp_c
            limits["components"] = comp_limits
        config["limits"] = limits

    return config


def _add_common_job_flags(parser: argparse.ArgumentParser) -> None:
    """Attach common simulation job override flags to a subcommand parser."""
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Path to JSON job configuration file (CLI flags override file values).",
    )
    parser.add_argument(
        "--pad",
        action="append",
        metavar="SPEC",
        help="Heat source pad spec REF.PAD=VALUE (e.g. U1.37=5W or U1.37=pwl.csv).",
    )
    parser.add_argument(
        "--current-flow",
        action="append",
        metavar="SPEC",
        help="Pad-to-pad DC current flow spec FROM->TO (e.g. 'J1.1->U1.1:3,U2.1:3,U3.1:3' or 'J1.1->U1.1,U1.2:12A').",
    )
    parser.add_argument(
        "--no-electro-thermal",
        dest="electro_thermal_coupling",
        action="store_false",
        default=None,
        help="Disable iterative rho_cu(T) electro-thermal coupling when current heating is enabled.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--steady-state",
        dest="steady_state",
        action="store_true",
        default=None,
        help="Run direct steady-state thermal solve.",
    )
    mode_group.add_argument(
        "--transient",
        dest="steady_state",
        action="store_false",
        default=None,
        help="Run transient BDF2 thermal solve.",
    )

    parser.add_argument("--time", type=_finite_float, help="Transient duration in seconds.")
    parser.add_argument("--amb", type=_finite_float, help="Ambient temperature in C.")
    parser.add_argument("--res", type=_finite_float, help="Target grid resolution in mm.")
    parser.add_argument("--thick", type=_finite_float, help="Fallback board thickness in mm.")
    parser.add_argument(
        "--h-conv",
        type=_finite_float,
        help="Top (and default bottom) effective convection coefficient in W/(m^2*K).",
    )
    parser.add_argument(
        "--h-conv-bottom",
        type=_finite_float,
        help="Bottom effective convection coefficient in W/(m^2*K) (defaults to --h-conv).",
    )
    parser.add_argument(
        "--via-plating-um",
        type=_finite_float,
        help="Via and PTH barrel copper plating thickness in um (5.0-100.0, default 25.0).",
    )
    parser.add_argument(
        "--via-fill",
        choices=["plated", "copper"],
        help="Via fill mode ('plated' or 'copper', default 'plated').",
    )
    parser.add_argument(
        "--detail",
        choices=["fast", "balanced", "detailed", "very_detailed", "custom"],
        help="Grid detail preset.",
    )
    parser.add_argument("--node-budget", type=int, help="Maximum grid node budget.")
    parser.add_argument(
        "--mesh",
        choices=["auto", "uniform", "adaptive"],
        help="Deprecated compatibility flag (ThermalSim v0.4.7+ always uses uniform meshing).",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "fast_native", "legacy"],
        help="Compute engine selection.",
    )
    parser.add_argument(
        "--use-heatsink",
        action="store_true",
        default=None,
        help="Enable Eco1 thermal pad / heatsink boundary.",
    )
    parser.add_argument(
        "--sink-mode",
        choices=["fixed", "heatsink"],
        help="Thermal pad sink mode.",
    )
    parser.add_argument("--heatsink-rth", type=_finite_float, help="Heatsink R_th in K/W.")
    parser.add_argument("--heatsink-mass", type=_finite_float, help="Heatsink mass in grams.")
    parser.add_argument(
        "--heatsink-material",
        choices=["aluminium", "copper", "steel", "magnesium", "custom"],
        help="Heatsink material preset.",
    )
    parser.add_argument("--heatsink-cp", type=_finite_float, help="Heatsink specific heat in J/(kg*K).")
    parser.add_argument("--pad-k", type=_finite_float, help="Thermal pad conductivity in W/(m*K).")
    parser.add_argument("--pad-th", type=_finite_float, help="Thermal pad thickness in mm.")
    parser.add_argument("--contact-factor", type=_finite_float, help="Thermal pad contact factor (0..1).")
    parser.add_argument("--sink-temp", type=_finite_float, help="Fixed sink temperature in C.")
    parser.add_argument("--limit-board", type=_finite_float, help="Max board temperature limit in C.")
    parser.add_argument("--limit-heatsink", type=_finite_float, help="Max heatsink temperature limit in C.")
    parser.add_argument(
        "--limit-comp",
        action="append",
        metavar="SPEC",
        help="Component temperature limit PATTERN=TEMP_C (e.g. C*=105).",
    )
    parser.add_argument("--power-scale", type=_finite_float, help="Scale factor for constant pad powers.")
    parser.add_argument(
        "--copper-thickness-scale",
        type=_finite_float,
        help="Scale factor for stackup copper thickness.",
    )
    parser.add_argument(
        "--refill-zones",
        action="store_true",
        default=None,
        help="Attempt to refill unfilled copper zones before solving.",
    )


def _add_pretty_flag(parser: argparse.ArgumentParser) -> None:
    """Add ``--pretty`` with SUPPRESS default so top-level or subcommand placement works."""
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Pretty-print JSON output with indentation.",
    )


def _build_parser() -> _JsonArgumentParser:
    """Construct the CLI argument parser with all subcommands."""
    parser = _JsonArgumentParser(
        prog="thermalsim",
        description="Headless ThermalSim CLI for PCB thermal inspection, simulation, and sweeps.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="Pretty-print JSON output with indentation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_inspect = subparsers.add_parser("inspect", help="Inspect board stackup, footprints, pads, and zones.")
    p_inspect.add_argument("board", help="Path to .kicad_pcb file.")
    _add_pretty_flag(p_inspect)

    p_validate = subparsers.add_parser("validate", help="Validate board and job configuration without solving.")
    p_validate.add_argument("board", help="Path to .kicad_pcb file.")
    _add_common_job_flags(p_validate)
    _add_pretty_flag(p_validate)

    p_run = subparsers.add_parser("run", help="Run headless thermal simulation and emit semantic summary JSON.")
    p_run.add_argument("board", help="Path to .kicad_pcb file.")
    p_run.add_argument("--out", metavar="DIR", default=None, help="Output directory for summary/artifacts.")
    _add_common_job_flags(p_run)
    p_run.add_argument("--no-plots", action="store_true", help="Skip generating PNG plots.")
    p_run.add_argument("--no-html", action="store_true", help="Skip generating HTML report.")
    _add_pretty_flag(p_run)

    p_sweep = subparsers.add_parser("sweep", help="Run 1D parameter sweep reusing geometry state.")
    p_sweep.add_argument("board", help="Path to .kicad_pcb file.")
    p_sweep.add_argument("--param", required=True, metavar="NAME", help="Parameter name to sweep.")
    p_sweep.add_argument("--values", required=True, metavar="V1,V2,...", help="Comma-separated sweep values.")
    p_sweep.add_argument("--out", metavar="DIR", default=None, help="Output directory for sweep summary/runs.")
    _add_common_job_flags(p_sweep)
    p_sweep.add_argument("--plots", action="store_true", help="Generate PNG plots for each sweep point.")
    _add_pretty_flag(p_sweep)

    p_compare = subparsers.add_parser("compare", help="Compare two simulation summary runs (B - A).")
    p_compare.add_argument("run_a", help="Baseline run directory or summary.json path.")
    p_compare.add_argument("run_b", help="Candidate run directory or summary.json path.")
    _add_pretty_flag(p_compare)

    return parser


def _determine_exit_code(command: str, result: Dict[str, Any]) -> int:
    """Map headless_api JSON result to CLI exit code (0, 1, 2, or 3)."""
    status = result.get("status")
    if status == "validation_error":
        return 2
    if command == "validate" and result.get("valid") is False:
        return 2
    if status == "limit_violation":
        return 3
    limits_check = result.get("limits_check")
    if isinstance(limits_check, dict) and limits_check.get("passed") is False:
        return 3
    if status == "error":
        return 1
    return 0


def _write_json(payload: Dict[str, Any], pretty: bool) -> None:
    """Serialize payload strictly to stdout as a single JSON document."""
    text = json.dumps(payload, indent=2 if pretty else None, allow_nan=False)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def main(argv: Optional[List[str]] = None) -> int:
    """Execute the headless ThermalSim CLI and return an exit code."""
    os.environ.setdefault("THERMALSIM_HEADLESS", "1")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    pretty = "--pretty" in raw_argv
    command_hint: Optional[str] = None

    try:
        parser = _build_parser()
        args = parser.parse_args(raw_argv)
        pretty = bool(getattr(args, "pretty", pretty))
        command_hint = getattr(args, "command", None)

        if args.command == "inspect":
            api = _get_headless_api()
            with contextlib.redirect_stdout(sys.stderr):
                result = api.inspect_board(args.board)
        elif args.command == "validate":
            job_config = _build_job_config(args)
            api = _get_headless_api()
            with contextlib.redirect_stdout(sys.stderr):
                result = api.validate_job(args.board, job_config)
        elif args.command == "run":
            job_config = _build_job_config(args)
            refill = bool(job_config.get("refill_zones", False))
            api = _get_headless_api()
            with contextlib.redirect_stdout(sys.stderr):
                result = api.run_headless_simulation(
                    args.board,
                    job_config,
                    out_dir=args.out,
                    generate_plots=not args.no_plots,
                    generate_html=not args.no_html,
                    refill_zones=refill,
                )
        elif args.command == "sweep":
            values = _parse_values_list(args.values)
            job_config = _build_job_config(args)
            refill = bool(job_config.get("refill_zones", False))
            api = _get_headless_api()
            with contextlib.redirect_stdout(sys.stderr):
                result = api.run_parameter_sweep(
                    args.board,
                    job_config,
                    args.param,
                    values,
                    out_dir=args.out,
                    generate_plots=bool(args.plots),
                    refill_zones=refill,
                )
        elif args.command == "compare":
            api = _get_headless_api()
            with contextlib.redirect_stdout(sys.stderr):
                result = api.compare_runs(args.run_a, args.run_b)
        else:
            raise _CliUsageError(f"Unknown command: {args.command}")

        _write_json(result, pretty=pretty)
        return _determine_exit_code(args.command, result)

    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 1

    except _CliUsageError as exc:
        err_payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "validation_error",
            "valid": False,
            "errors": [{"code": "CLI_ERROR", "message": str(exc)}],
        }
        if command_hint:
            err_payload["command"] = command_hint
        _write_json(err_payload, pretty=pretty)
        return 2

    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        err_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": {
                "code": "RUNTIME_ERROR",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        if command_hint:
            err_payload["command"] = command_hint
        _write_json(err_payload, pretty=pretty)
        return 1


if __name__ == "__main__":
    sys.exit(main())
