"""Project-scoped heat-source and current-path configuration helpers.

The GUI deliberately works with high-level source and path definitions.  This
module turns those definitions into small, serialisable specifications that a
controller can resolve to live KiCad pads immediately before it invokes the
thermal or electrical solvers.  It has no :mod:`pcbnew` dependency so it is
also safe to use for persistence and preflight work.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
import uuid


SCHEMA_VERSION = 3
"""Current on-disk schema version for project source configuration."""


def _text(value: Any, default: str = "") -> str:
    """Return a trimmed string while keeping ``None`` out of persisted data."""
    if value is None:
        return default
    return str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    """Return a finite float, or ``default`` for malformed input."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _positive_or_none(value: Any) -> Optional[float]:
    """Return a finite positive float, or ``None`` when no usable value exists."""
    result = _float(value, 0.0)
    return result if result > 0.0 else None


def _bool(value: Any, default: bool = False) -> bool:
    """Coerce common JSON and UI boolean forms without treating ``'false'`` as true."""
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _mapping(value: Any) -> Dict[str, Any]:
    """Return a shallow plain dictionary for mapping-like JSON values."""
    return dict(value) if isinstance(value, Mapping) else {}


def _copy_json_mapping(value: Any) -> Dict[str, Any]:
    """Make a defensive copy of JSON-shaped metadata without requiring JSON round trips."""
    return dict(value) if isinstance(value, Mapping) else {}


def _unique_texts(values: Iterable[Any]) -> List[str]:
    """Return non-empty text values once, preserving their first-seen order."""
    result: List[str] = []
    seen = set()
    for value in values:
        text = _text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _normalise_distribution(value: Any) -> str:
    """Map accepted UI aliases to the three supported pad-share modes."""
    normalized = _text(value, "area").lower().replace("-", "_").replace(" ", "_")
    if normalized in {"equal", "uniform", "even", "split_evenly"}:
        return "equal"
    if normalized in {"custom", "manual", "shares", "advanced"}:
        return "custom"
    return "area"


def _normalise_shares(values: Optional[Sequence[Any]], count: int) -> List[float]:
    """Return non-negative shares that sum to one, falling back to equal shares."""
    if count <= 0:
        return []
    if values is None or len(values) != count:
        return [1.0 / float(count)] * count

    raw = [_float(value, float("nan")) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in raw):
        return [1.0 / float(count)] * count
    total = math.fsum(raw)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / float(count)] * count

    normalized = [value / total for value in raw]
    # Make the documented invariant exact for math.fsum users and avoid an
    # accumulated rounding error in a downstream power/current total.
    normalized[-1] = 1.0 - math.fsum(normalized[:-1])
    if normalized[-1] < 0.0:
        return [1.0 / float(count)] * count
    return normalized


def _stable_id(prefix: str, *parts: Any) -> str:
    """Build a deterministic opaque identifier for migrated legacy objects."""
    seed = "|".join(_text(part) for part in parts)
    return "%s-%s" % (prefix, uuid.uuid5(uuid.NAMESPACE_URL, "thermalsim/v3/" + seed).hex)


@dataclass
class PadRef:
    """Serializable stable reference to a KiCad pad.

    Parameters
    ----------
    pad_uuid : str
        KiCad pad UUID, preferred when resolving a live pad.
    footprint_uuid : str
        KiCad footprint UUID used with ``pad_number`` as a fallback.
    reference : str
        Human-readable footprint reference, such as ``U1``.
    pad_number : str
        KiCad pad number/name within the footprint.
    display_name : str
        UI label retained independently from the stable identity.
    net_name, net_code : str, int
        Last-known net display data.  These are not pad identity.
    layer : str
        Last-known primary pad layer label.
    pin_function : str
        Optional pin-function label for searching and display.
    area_mm2 : float, optional
        Last-known effective copper contact area for area-weighted splitting.
    legacy_key : str
        Previous schema's positional pad key retained as a final compatibility
        lookup hint; new code must not make it the primary identity.
    """

    pad_uuid: str = ""
    footprint_uuid: str = ""
    reference: str = ""
    pad_number: str = ""
    display_name: str = ""
    net_name: str = ""
    net_code: int = 0
    layer: str = ""
    pin_function: str = ""
    area_mm2: Optional[float] = None
    legacy_key: str = ""

    def normalized(self) -> "PadRef":
        """Return this pad reference with stable, JSON-safe field values.

        Returns
        -------
        PadRef
            A new normalised reference.  Non-positive or invalid areas become
            ``None`` so area distribution can fall back to equal splitting.
        """
        return PadRef(
            pad_uuid=_text(self.pad_uuid),
            footprint_uuid=_text(self.footprint_uuid),
            reference=_text(self.reference),
            pad_number=_text(self.pad_number),
            display_name=_text(self.display_name),
            net_name=_text(self.net_name),
            net_code=int(_float(self.net_code, 0.0)),
            layer=_text(self.layer),
            pin_function=_text(self.pin_function),
            area_mm2=_positive_or_none(self.area_mm2),
            legacy_key=_text(self.legacy_key),
        )

    def identity_key(self) -> str:
        """Return the preferred serialisable key for share maps and lookups.

        Returns
        -------
        str
            A key ordered by pad UUID, footprint UUID plus pad number,
            reference plus pad number, then the legacy positional key.
        """
        pad = self.normalized()
        if pad.pad_uuid:
            return "pad:" + pad.pad_uuid
        if pad.footprint_uuid and pad.pad_number:
            return "footprint:%s/pad:%s" % (pad.footprint_uuid, pad.pad_number)
        if pad.reference or pad.pad_number:
            return "reference:%s/pad:%s" % (pad.reference, pad.pad_number)
        if pad.legacy_key:
            return "legacy:" + pad.legacy_key
        return "display:" + pad.display_name

    def has_locator(self) -> bool:
        """Return whether this reference contains at least one usable pad locator.

        Returns
        -------
        bool
            ``True`` when a UUID, reference/number pair, legacy key, or display
            label can guide a repair workflow.
        """
        pad = self.normalized()
        return bool(
            pad.pad_uuid
            or (pad.footprint_uuid and pad.pad_number)
            or (pad.reference and pad.pad_number)
            or pad.legacy_key
            or pad.display_name
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this pad reference to schema-v3 JSON primitives.

        Returns
        -------
        dict
            A plain dictionary suitable for :func:`json.dump`.
        """
        pad = self.normalized()
        return {
            "pad_uuid": pad.pad_uuid,
            "footprint_uuid": pad.footprint_uuid,
            "reference": pad.reference,
            "pad_number": pad.pad_number,
            "display_name": pad.display_name,
            "net_name": pad.net_name,
            "net_code": pad.net_code,
            "layer": pad.layer,
            "pin_function": pad.pin_function,
            "area_mm2": pad.area_mm2,
            "legacy_key": pad.legacy_key,
        }

    @classmethod
    def from_dict(cls, data: Union["PadRef", Mapping[str, Any]]) -> "PadRef":
        """Construct a normalised pad reference from v3 or legacy descriptor data.

        Parameters
        ----------
        data : PadRef or mapping
            A v3 pad object or an earlier ``power_pads``/``current_groups``
            descriptor.

        Returns
        -------
        PadRef
            A normalised, serialisable pad reference.
        """
        if isinstance(data, PadRef):
            return data.normalized()
        raw = _mapping(data)
        legacy_key = _text(raw.get("legacy_key", raw.get("pad_key", raw.get("key", ""))))
        display_name = _text(raw.get("display_name", raw.get("name", "")))
        reference = _text(raw.get("reference", raw.get("ref", raw.get("footprint_ref", ""))))
        pad_number = _text(raw.get("pad_number", raw.get("number", raw.get("pad", ""))))

        # A v2 key was ``reference:pad:net-code:x:y``.  It is only a fallback
        # identity, but it gives us a useful reference/pad repair target.
        pieces = legacy_key.split(":")
        if (not reference or not pad_number) and len(pieces) >= 2:
            reference = reference or _text(pieces[0])
            pad_number = pad_number or _text(pieces[1])
        if (not reference or not pad_number) and "-" in display_name:
            label = display_name.split("[", 1)[0].strip()
            possible_reference, possible_pad = label.rsplit("-", 1)
            reference = reference or _text(possible_reference)
            pad_number = pad_number or _text(possible_pad)

        return cls(
            pad_uuid=_text(raw.get("pad_uuid", raw.get("uuid", raw.get("pad_id", "")))),
            footprint_uuid=_text(raw.get("footprint_uuid", raw.get("footprint_id", ""))),
            reference=reference,
            pad_number=pad_number,
            display_name=display_name,
            net_name=_text(raw.get("net_name", raw.get("net", ""))),
            net_code=int(_float(raw.get("net_code", 0), 0.0)),
            layer=_text(raw.get("layer", "")),
            pin_function=_text(raw.get("pin_function", raw.get("function", ""))),
            area_mm2=_positive_or_none(raw.get("area_mm2", raw.get("area", None))),
            legacy_key=legacy_key,
        ).normalized()


def _shares_from_data(value: Any, pads: Sequence[PadRef]) -> List[Any]:
    """Return ordered share values from either a list or a pad-keyed mapping."""
    if isinstance(value, Mapping):
        result = []
        for pad in pads:
            candidates = (
                pad.identity_key(),
                pad.pad_uuid,
                pad.legacy_key,
                "%s-%s" % (pad.reference, pad.pad_number),
                pad.display_name,
            )
            found = 0.0
            for candidate in candidates:
                if candidate and candidate in value:
                    found = value[candidate]
                    break
            result.append(found)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


@dataclass
class HeatSourceDefinition:
    """A total heat-loss profile distributed across one or more pads.

    ``power_w`` is used for ``profile_kind='constant'``.  For
    ``profile_kind='pwl'``, ``pwl_path`` points to a total-power PWL file; the
    expansion helper applies each pad's share rather than treating the PWL as a
    per-pad value.
    """

    id: str = ""
    name: str = ""
    profile_kind: str = "constant"
    power_w: float = 0.0
    pwl_path: str = ""
    distribution: str = "area"
    pads: List[PadRef] = field(default_factory=list)
    custom_shares: List[float] = field(default_factory=list)
    enabled: bool = True
    needs_repair: bool = False
    repair_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "HeatSourceDefinition":
        """Return a schema-safe copy with canonical profile and share fields.

        Returns
        -------
        HeatSourceDefinition
            A new definition with only ``constant`` or ``pwl`` profiles and
            custom shares normalised to the selected pad count.
        """
        pads = [PadRef.from_dict(pad) for pad in (self.pads or [])]
        kind = _text(self.profile_kind, "constant").lower()
        if kind not in {"constant", "pwl"}:
            kind = "constant"
        return HeatSourceDefinition(
            id=_text(self.id),
            name=_text(self.name),
            profile_kind=kind,
            power_w=_float(self.power_w, 0.0),
            pwl_path=_text(self.pwl_path),
            distribution=_normalise_distribution(self.distribution),
            pads=pads,
            custom_shares=_normalise_shares(list(self.custom_shares or []), len(pads)),
            enabled=_bool(self.enabled, True),
            needs_repair=_bool(self.needs_repair, False),
            repair_reason=_text(self.repair_reason),
            metadata=_copy_json_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this heat source using the schema-v3 total-profile shape.

        Returns
        -------
        dict
            Plain JSON-compatible heat-source data.
        """
        source = self.normalized()
        profile: Dict[str, Any]
        if source.profile_kind == "pwl":
            profile = {"kind": "pwl", "path": source.pwl_path}
        else:
            profile = {"kind": "constant", "value_w": source.power_w}
        return {
            "id": source.id,
            "name": source.name,
            "enabled": source.enabled,
            "needs_repair": source.needs_repair,
            "repair_reason": source.repair_reason,
            "power_profile": profile,
            "distribution": source.distribution,
            "pads": [pad.to_dict() for pad in source.pads],
            "custom_shares": source.custom_shares,
            "metadata": source.metadata,
        }

    @classmethod
    def from_dict(
        cls, data: Union["HeatSourceDefinition", Mapping[str, Any]]
    ) -> "HeatSourceDefinition":
        """Construct a heat-source definition from JSON-compatible data.

        Parameters
        ----------
        data : HeatSourceDefinition or mapping
            A schema-v3 object.  Legacy scalar fields are accepted to make
            callers that progressively migrate settings straightforward.

        Returns
        -------
        HeatSourceDefinition
            A normalised total-power heat-source definition.
        """
        if isinstance(data, HeatSourceDefinition):
            return data.normalized()
        raw = _mapping(data)
        profile = _mapping(raw.get("power_profile", raw.get("profile", {})))
        kind = _text(profile.get("kind", raw.get("profile_kind", "constant")), "constant").lower()
        if kind not in {"constant", "pwl"}:
            kind = "constant"
        pads = [PadRef.from_dict(pad) for pad in raw.get("pads", raw.get("pad_refs", [])) or []]
        shares = _shares_from_data(
            raw.get("custom_shares", raw.get("pad_shares", raw.get("shares", []))), pads
        )
        return cls(
            id=_text(raw.get("id", raw.get("source_id", ""))),
            name=_text(raw.get("name", "")),
            profile_kind=kind,
            power_w=_float(
                profile.get("value_w", profile.get("power_w", raw.get("power_w", raw.get("power", 0.0)))),
                0.0,
            ),
            pwl_path=_text(profile.get("path", raw.get("pwl_path", raw.get("path", "")))),
            distribution=_normalise_distribution(raw.get("distribution", "area")),
            pads=pads,
            custom_shares=shares,
            enabled=_bool(raw.get("enabled", True), True),
            needs_repair=_bool(raw.get("needs_repair", False), False),
            repair_reason=_text(raw.get("repair_reason", "")),
            metadata=_copy_json_mapping(raw.get("metadata", {})),
        ).normalized()


@dataclass
class CurrentPathDefinition:
    """A balanced current path on one copper net.

    Positive source terminals inject ``current_a`` into the named net and
    negative sink terminals remove the same magnitude.  A non-empty
    ``circuit_id`` makes the corresponding :class:`CurrentCircuitDefinition`
    current authoritative during expansion.
    """

    id: str = ""
    name: str = ""
    net_name: str = ""
    net_code: int = 0
    current_a: float = 0.0
    source_pads: List[PadRef] = field(default_factory=list)
    sink_pads: List[PadRef] = field(default_factory=list)
    source_shares: List[float] = field(default_factory=list)
    sink_shares: List[float] = field(default_factory=list)
    circuit_id: str = ""
    enabled: bool = True
    needs_repair: bool = False
    repair_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "CurrentPathDefinition":
        """Return a canonical copy with non-negative current and normalised shares.

        Returns
        -------
        CurrentPathDefinition
            A new path definition.  Source and sink shares independently sum
            to one whenever their respective pad list is non-empty.
        """
        source_pads = [PadRef.from_dict(pad) for pad in (self.source_pads or [])]
        sink_pads = [PadRef.from_dict(pad) for pad in (self.sink_pads or [])]
        return CurrentPathDefinition(
            id=_text(self.id),
            name=_text(self.name),
            net_name=_text(self.net_name),
            net_code=int(_float(self.net_code, 0.0)),
            current_a=abs(_float(self.current_a, 0.0)),
            source_pads=source_pads,
            sink_pads=sink_pads,
            source_shares=_normalise_shares(list(self.source_shares or []), len(source_pads)),
            sink_shares=_normalise_shares(list(self.sink_shares or []), len(sink_pads)),
            circuit_id=_text(self.circuit_id),
            enabled=_bool(self.enabled, True),
            needs_repair=_bool(self.needs_repair, False),
            repair_reason=_text(self.repair_reason),
            metadata=_copy_json_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this current path to schema-v3 JSON primitives.

        Returns
        -------
        dict
            A plain path object with ordered source/sink pad shares.
        """
        path = self.normalized()
        return {
            "id": path.id,
            "name": path.name,
            "net_name": path.net_name,
            "net_code": path.net_code,
            "current_a": path.current_a,
            "source_pads": [pad.to_dict() for pad in path.source_pads],
            "sink_pads": [pad.to_dict() for pad in path.sink_pads],
            "source_shares": path.source_shares,
            "sink_shares": path.sink_shares,
            "circuit_id": path.circuit_id,
            "enabled": path.enabled,
            "needs_repair": path.needs_repair,
            "repair_reason": path.repair_reason,
            "metadata": path.metadata,
        }

    @classmethod
    def from_dict(
        cls, data: Union["CurrentPathDefinition", Mapping[str, Any]]
    ) -> "CurrentPathDefinition":
        """Construct a current path from a v3 JSON object.

        Parameters
        ----------
        data : CurrentPathDefinition or mapping
            A path object.  ``sources``/``sinks`` aliases are accepted for UI
            callers before serialisation.

        Returns
        -------
        CurrentPathDefinition
            A normalised path definition.
        """
        if isinstance(data, CurrentPathDefinition):
            return data.normalized()
        raw = _mapping(data)
        source_pads = [
            PadRef.from_dict(pad)
            for pad in raw.get("source_pads", raw.get("sources", [])) or []
        ]
        sink_pads = [
            PadRef.from_dict(pad)
            for pad in raw.get("sink_pads", raw.get("sinks", [])) or []
        ]
        net_name = _text(raw.get("net_name", raw.get("net", "")))
        if not net_name:
            for pad in source_pads + sink_pads:
                if pad.net_name:
                    net_name = pad.net_name
                    break
        return cls(
            id=_text(raw.get("id", raw.get("path_id", ""))),
            name=_text(raw.get("name", "")),
            net_name=net_name,
            net_code=int(_float(raw.get("net_code", 0), 0.0)),
            current_a=abs(_float(raw.get("current_a", raw.get("current", 0.0)), 0.0)),
            source_pads=source_pads,
            sink_pads=sink_pads,
            source_shares=_shares_from_data(
                raw.get("source_shares", raw.get("source_weights", [])), source_pads
            ),
            sink_shares=_shares_from_data(
                raw.get("sink_shares", raw.get("sink_weights", [])), sink_pads
            ),
            circuit_id=_text(raw.get("circuit_id", raw.get("circuit", ""))),
            enabled=_bool(raw.get("enabled", True), True),
            needs_repair=_bool(raw.get("needs_repair", False), False),
            repair_reason=_text(raw.get("repair_reason", "")),
            metadata=_copy_json_mapping(raw.get("metadata", {})),
        ).normalized()


@dataclass
class CurrentCircuitDefinition:
    """Optional DC circuit that supplies a common current to linked net paths.

    Both the path's ``circuit_id`` and this object's ``path_ids`` are stored so
    a GUI can navigate the relationship from either side.  Normalisation keeps
    the two lists consistent when possible.
    """

    id: str = ""
    name: str = ""
    current_a: float = 0.0
    path_ids: List[str] = field(default_factory=list)
    enabled: bool = True
    needs_repair: bool = False
    repair_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "CurrentCircuitDefinition":
        """Return a canonical copy of this DC-circuit definition.

        Returns
        -------
        CurrentCircuitDefinition
            A copy with an absolute current magnitude and duplicate-free path
            identifiers.
        """
        return CurrentCircuitDefinition(
            id=_text(self.id),
            name=_text(self.name),
            current_a=abs(_float(self.current_a, 0.0)),
            path_ids=_unique_texts(self.path_ids or []),
            enabled=_bool(self.enabled, True),
            needs_repair=_bool(self.needs_repair, False),
            repair_reason=_text(self.repair_reason),
            metadata=_copy_json_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this DC circuit to schema-v3 JSON primitives.

        Returns
        -------
        dict
            A plain JSON-compatible circuit object.
        """
        circuit = self.normalized()
        return {
            "id": circuit.id,
            "name": circuit.name,
            "current_a": circuit.current_a,
            "path_ids": circuit.path_ids,
            "enabled": circuit.enabled,
            "needs_repair": circuit.needs_repair,
            "repair_reason": circuit.repair_reason,
            "metadata": circuit.metadata,
        }

    @classmethod
    def from_dict(
        cls, data: Union["CurrentCircuitDefinition", Mapping[str, Any]]
    ) -> "CurrentCircuitDefinition":
        """Construct a DC-circuit definition from JSON-compatible data.

        Parameters
        ----------
        data : CurrentCircuitDefinition or mapping
            A v3 circuit object.  ``paths`` is accepted as an input alias for
            ``path_ids``.

        Returns
        -------
        CurrentCircuitDefinition
            A normalised circuit definition.
        """
        if isinstance(data, CurrentCircuitDefinition):
            return data.normalized()
        raw = _mapping(data)
        return cls(
            id=_text(raw.get("id", raw.get("circuit_id", ""))),
            name=_text(raw.get("name", "")),
            current_a=abs(_float(raw.get("current_a", raw.get("current", 0.0)), 0.0)),
            path_ids=_unique_texts(raw.get("path_ids", raw.get("paths", [])) or []),
            enabled=_bool(raw.get("enabled", True), True),
            needs_repair=_bool(raw.get("needs_repair", False), False),
            repair_reason=_text(raw.get("repair_reason", "")),
            metadata=_copy_json_mapping(raw.get("metadata", {})),
        ).normalized()


def _assign_missing_ids(items: Sequence[Any], prefix: str) -> List[Any]:
    """Assign deterministic display-safe IDs to blank or duplicate definition IDs."""
    result: List[Any] = []
    seen = set()
    for index, item in enumerate(items, start=1):
        preferred = _text(getattr(item, "id", "")) or "%s-%d" % (prefix, index)
        candidate = preferred
        duplicate = 2
        while candidate in seen:
            candidate = "%s-%d" % (preferred, duplicate)
            duplicate += 1
        seen.add(candidate)
        result.append(replace(item, id=candidate))
    return result


@dataclass
class ProjectConfiguration:
    """Complete project-side source configuration stored in a schema-v3 sidecar.

    The ``extra`` mapping preserves unknown or migrated settings without making
    them part of the solver-facing source/path interface.  A controller can
    keep user/machine preferences elsewhere while still avoiding data loss on
    a first migration save.
    """

    heat_sources: List[HeatSourceDefinition] = field(default_factory=list)
    current_paths: List[CurrentPathDefinition] = field(default_factory=list)
    current_circuits: List[CurrentCircuitDefinition] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def normalized(self) -> "ProjectConfiguration":
        """Return a canonical schema-v3 configuration with linked circuit IDs.

        Returns
        -------
        ProjectConfiguration
            A copy whose definition IDs are non-empty/unique and whose
            circuit-to-path relationship is represented in both directions.
        """
        sources = _assign_missing_ids(
            [HeatSourceDefinition.from_dict(source) for source in (self.heat_sources or [])],
            "heat-source",
        )
        paths = _assign_missing_ids(
            [CurrentPathDefinition.from_dict(path) for path in (self.current_paths or [])],
            "current-path",
        )
        circuits = _assign_missing_ids(
            [CurrentCircuitDefinition.from_dict(circuit) for circuit in (self.current_circuits or [])],
            "current-circuit",
        )

        path_index = {path.id: index for index, path in enumerate(paths)}
        circuit_index = {circuit.id: index for index, circuit in enumerate(circuits)}
        circuit_paths = {circuit.id: list(circuit.path_ids) for circuit in circuits}

        # A path's explicit circuit link is authoritative; circuit path lists
        # fill in omitted path links for hand-authored JSON.
        for path in paths:
            if path.circuit_id in circuit_index and path.id not in circuit_paths[path.circuit_id]:
                circuit_paths[path.circuit_id].append(path.id)
        for circuit_id, path_ids in circuit_paths.items():
            for path_id in path_ids:
                index = path_index.get(path_id)
                if index is not None and not paths[index].circuit_id:
                    paths[index] = replace(paths[index], circuit_id=circuit_id)

        circuits = [
            replace(circuit, path_ids=_unique_texts(circuit_paths[circuit.id]))
            for circuit in circuits
        ]
        return ProjectConfiguration(
            heat_sources=sources,
            current_paths=paths,
            current_circuits=circuits,
            extra=_copy_json_mapping(self.extra),
            schema_version=SCHEMA_VERSION,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this complete configuration to schema-v3 JSON primitives.

        Returns
        -------
        dict
            A dictionary headed by ``schema_version: 3``.
        """
        config = self.normalized()
        data: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "heat_sources": [source.to_dict() for source in config.heat_sources],
            "current_paths": [path.to_dict() for path in config.current_paths],
            "current_circuits": [circuit.to_dict() for circuit in config.current_circuits],
        }
        if config.extra:
            data["extra"] = config.extra
        return data

    @classmethod
    def from_dict(
        cls, data: Union["ProjectConfiguration", Mapping[str, Any]]
    ) -> "ProjectConfiguration":
        """Construct a project configuration from schema-v3 JSON data.

        Parameters
        ----------
        data : ProjectConfiguration or mapping
            A v3 configuration object.  V2 callers should use
            :func:`migrate_v2_settings` explicitly or :func:`load_project_config`.

        Returns
        -------
        ProjectConfiguration
            A normalised schema-v3 configuration.
        """
        if isinstance(data, ProjectConfiguration):
            return data.normalized()
        raw = _mapping(data)
        known = {"schema_version", "heat_sources", "current_paths", "current_circuits", "extra"}
        extra = _copy_json_mapping(raw.get("extra", {}))
        for key, value in raw.items():
            if key not in known and key not in extra:
                extra[key] = value
        return cls(
            heat_sources=[
                HeatSourceDefinition.from_dict(source)
                for source in raw.get("heat_sources", raw.get("sources", [])) or []
            ],
            current_paths=[
                CurrentPathDefinition.from_dict(path)
                for path in raw.get("current_paths", raw.get("paths", [])) or []
            ],
            current_circuits=[
                CurrentCircuitDefinition.from_dict(circuit)
                for circuit in raw.get("current_circuits", raw.get("circuits", [])) or []
            ],
            extra=extra,
            schema_version=SCHEMA_VERSION,
        ).normalized()


def _coerce_configuration(
    config: Union[ProjectConfiguration, Mapping[str, Any]]
) -> ProjectConfiguration:
    """Coerce a v3 configuration or a legacy settings map into a project configuration."""
    if isinstance(config, ProjectConfiguration):
        return config.normalized()
    if not isinstance(config, Mapping):
        raise TypeError("config must be a ProjectConfiguration or mapping")
    version = int(_float(config.get("schema_version", 0), 0.0))
    if version <= 2 or "power_pads" in config or "current_groups" in config:
        return migrate_v2_settings(config)
    return ProjectConfiguration.from_dict(config)


def _with_absolute_pwl_paths(config: ProjectConfiguration, parent: Path) -> ProjectConfiguration:
    """Return a copy whose PWL paths are resolved from the sidecar directory."""
    sources: List[HeatSourceDefinition] = []
    for source in config.heat_sources:
        source = source.normalized()
        if source.profile_kind == "pwl" and source.pwl_path:
            raw_path = Path(source.pwl_path)
            if not raw_path.is_absolute():
                raw_path = parent / raw_path
            source = replace(source, pwl_path=os.path.abspath(os.fspath(raw_path)))
        sources.append(source)
    return replace(config, heat_sources=sources).normalized()


def _with_relative_pwl_paths(config: ProjectConfiguration, parent: Path) -> ProjectConfiguration:
    """Return a copy with PWL paths made relative to a sidecar directory where possible."""
    sources: List[HeatSourceDefinition] = []
    for source in config.heat_sources:
        source = source.normalized()
        if source.profile_kind == "pwl" and source.pwl_path:
            raw_path = Path(source.pwl_path)
            if not raw_path.is_absolute():
                raw_path = parent / raw_path
            absolute = os.path.abspath(os.fspath(raw_path))
            try:
                stored_path = os.path.relpath(absolute, os.fspath(parent))
            except ValueError:
                # Windows cannot make a relative path across drive letters.
                stored_path = absolute
            source = replace(source, pwl_path=stored_path)
        sources.append(source)
    return replace(config, heat_sources=sources).normalized()


def save_project_config(
    path: Union[str, os.PathLike], config: Union[ProjectConfiguration, Mapping[str, Any]]
) -> Path:
    """Atomically save schema-v3 project configuration to a JSON sidecar.

    PWL profile paths are stored relative to the sidecar directory whenever the
    operating system can express that relative path.  The caller's in-memory
    configuration is not mutated.

    Parameters
    ----------
    path : str or os.PathLike
        Destination sidecar path, normally ``<board-name>.thermalsim.json``.
    config : ProjectConfiguration or mapping
        A schema-v3 configuration or a v2 settings map to migrate before save.

    Returns
    -------
    pathlib.Path
        The resolved destination path after the atomic replace succeeds.

    Raises
    ------
    OSError
        If the sidecar cannot be written or atomically replaced.
    TypeError
        If ``config`` is neither a project configuration nor a mapping.
    """
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = Path(os.path.abspath(os.fspath(destination)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _with_relative_pwl_paths(_coerce_configuration(config), destination.parent).to_dict()

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination.name,
        suffix=".tmp",
        dir=os.fspath(destination.parent),
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, os.fspath(destination))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_project_config(path: Union[str, os.PathLike]) -> ProjectConfiguration:
    """Load a schema-v3 sidecar, automatically migrating a v2 settings file.

    Relative PWL paths in the JSON sidecar become absolute paths relative to
    the sidecar's parent directory.  This makes simulation execution
    independent of KiCad's current working directory.

    Parameters
    ----------
    path : str or os.PathLike
        Source sidecar path.

    Returns
    -------
    ProjectConfiguration
        A normalised schema-v3 configuration.

    Raises
    ------
    FileNotFoundError
        If the sidecar does not exist.
    ValueError
        If the JSON root is not an object or declares a newer schema.
    json.JSONDecodeError
        If the sidecar is malformed JSON.
    """
    source_path = Path(path).expanduser()
    if not source_path.is_absolute():
        source_path = Path(os.path.abspath(os.fspath(source_path)))
    with source_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("ThermalSim project configuration must be a JSON object")
    version = int(_float(raw.get("schema_version", 0), 0.0))
    if version > SCHEMA_VERSION:
        raise ValueError(
            "ThermalSim configuration schema %d is newer than supported schema %d"
            % (version, SCHEMA_VERSION)
        )
    if version <= 2 or "power_pads" in raw or "current_groups" in raw:
        config = migrate_v2_settings(raw)
    else:
        config = ProjectConfiguration.from_dict(raw)
    return _with_absolute_pwl_paths(config, source_path.parent)


def _legacy_power_value(raw_pad: Mapping[str, Any], fallback_values: Sequence[str], index: int) -> str:
    """Use the v2 per-pad value when present, otherwise apply legacy power_str semantics."""
    for key in ("power", "power_str", "power_w"):
        if key in raw_pad and raw_pad[key] is not None:
            return _text(raw_pad[key])
    if len(fallback_values) == 1:
        return fallback_values[0]
    if index < len(fallback_values):
        return fallback_values[index]
    return "0.0"


def _legacy_profile(value: str) -> Dict[str, Any]:
    """Translate an old power text field to a total constant or PWL profile."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return {"kind": "pwl", "path": _text(value)}
    if math.isfinite(numeric):
        return {"kind": "constant", "value_w": numeric}
    return {"kind": "constant", "value_w": 0.0}


def _legacy_group_currents(group: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Resolve v2 total/per-pad group data into individual signed pad currents."""
    raw_pads = [_mapping(pad) for pad in group.get("pads", []) or []]
    mode = _text(group.get("mode", "per_pad")).lower()
    is_total = mode in {"total", "gesamtstrom_verteilen", "gesamtstrom verteilen"}
    total = _float(group.get("total_current_a", group.get("total_current", 0.0)), 0.0)
    resolved: List[Dict[str, Any]] = []
    for raw_pad in raw_pads:
        current = total / float(len(raw_pads)) if is_total and raw_pads else _float(
            raw_pad.get("current_a", raw_pad.get("current", 0.0)), 0.0
        )
        resolved.append({"pad": PadRef.from_dict(raw_pad), "current_a": current})
    return resolved


def migrate_v2_settings(settings: Mapping[str, Any]) -> ProjectConfiguration:
    """Migrate v2 ``power_pads`` and ``current_groups`` to schema-v3 objects.

    Each v2 power pad becomes one independent total-power heat source.  V2
    terminal entries are grouped by copper net into current paths; their signed
    currents become source/sink shares.  A net that is missing either side or
    whose absolute source/sink totals differ is retained as a disabled,
    repair-required path and therefore can never execute silently.

    Parameters
    ----------
    settings : mapping
        Existing settings dictionary using schema version 2 or earlier.

    Returns
    -------
    ProjectConfiguration
        A normalised schema-v3 configuration.  Unrelated settings are retained
        under ``extra`` for a controller to migrate to its user profile.
    """
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")
    raw_settings = dict(settings)
    raw_power_pads = [_mapping(pad) for pad in raw_settings.get("power_pads", []) or []]
    fallback_values = [
        item.strip() for item in _text(raw_settings.get("power_str", "")).split(",") if item.strip()
    ]
    if not fallback_values:
        fallback_values = ["0.0"]

    heat_sources: List[HeatSourceDefinition] = []
    for index, raw_pad in enumerate(raw_power_pads):
        pad = PadRef.from_dict(raw_pad)
        power_text = _legacy_power_value(raw_pad, fallback_values, index)
        profile = _legacy_profile(power_text)
        source_name = _text(raw_pad.get("name", "")) or "Heat source %d" % (index + 1)
        repair_reason = "" if pad.has_locator() else "Legacy heat source has no pad reference."
        heat_sources.append(
            HeatSourceDefinition(
                id=_stable_id("heat-source", index, pad.identity_key()),
                name=source_name,
                profile_kind=profile["kind"],
                power_w=_float(profile.get("value_w", 0.0), 0.0),
                pwl_path=_text(profile.get("path", "")),
                distribution="equal",
                pads=[pad],
                custom_shares=[1.0],
                enabled=True,
                needs_repair=bool(repair_reason),
                repair_reason=repair_reason,
                metadata={"legacy_power": power_text},
            )
        )

    # Preserve the first-seen net order.  Keys include the net code only when
    # no display name exists, because a net code is explicitly not stable
    # across board edits.
    grouped: Dict[str, Dict[str, Any]] = {}
    for group_index, raw_group in enumerate(raw_settings.get("current_groups", []) or []):
        group = _mapping(raw_group)
        group_name = _text(group.get("name", "")) or "Group %d" % (group_index + 1)
        group_color = _text(group.get("color", ""))
        for entry in _legacy_group_currents(group):
            pad = entry["pad"]
            net_name = pad.net_name or _text(group.get("net_name", group.get("net", "")))
            net_code = pad.net_code or int(_float(group.get("net_code", 0), 0.0))
            net_key = net_name if net_name else "#%d" % net_code if net_code else "(unassigned)"
            bucket = grouped.setdefault(
                net_key,
                {
                    "net_name": net_name,
                    "net_code": net_code,
                    "sources": [],
                    "sinks": [],
                    "names": [],
                    "colors": [],
                },
            )
            bucket["names"].append(group_name)
            if group_color:
                bucket["colors"].append(group_color)
            current = _float(entry["current_a"], 0.0)
            if current > 0.0:
                bucket["sources"].append((pad, current))
            elif current < 0.0:
                bucket["sinks"].append((pad, abs(current)))

    paths: List[CurrentPathDefinition] = []
    current_enabled = _bool(raw_settings.get("current_enabled", False), False)
    for net_key, bucket in grouped.items():
        source_pads = [item[0] for item in bucket["sources"]]
        sink_pads = [item[0] for item in bucket["sinks"]]
        source_values = [item[1] for item in bucket["sources"]]
        sink_values = [item[1] for item in bucket["sinks"]]
        source_total = math.fsum(source_values)
        sink_total = math.fsum(sink_values)
        balanced = bool(source_pads and sink_pads) and math.isclose(
            source_total, sink_total, rel_tol=1.0e-6, abs_tol=1.0e-9
        )
        net_name = _text(bucket["net_name"])
        repair_reasons: List[str] = []
        if not net_name:
            repair_reasons.append("Legacy current path has no net name.")
        if not source_pads or not sink_pads:
            repair_reasons.append("Legacy current net needs both source and sink pads.")
        elif not balanced:
            repair_reasons.append(
                "Legacy current net '%s' is not current-balanced (sources %.12g A, sinks %.12g A)."
                % (net_name or net_key, source_total, sink_total)
            )
        needs_repair = bool(repair_reasons)
        path_names = _unique_texts(bucket["names"])
        path_name = path_names[0] if len(path_names) == 1 else (net_name or net_key) + " path"
        paths.append(
            CurrentPathDefinition(
                id=_stable_id("current-path", net_key),
                name=path_name,
                net_name=net_name,
                net_code=int(_float(bucket["net_code"], 0.0)),
                current_a=source_total if source_total else sink_total,
                source_pads=source_pads,
                sink_pads=sink_pads,
                source_shares=_normalise_shares(source_values, len(source_values)),
                sink_shares=_normalise_shares(sink_values, len(sink_values)),
                enabled=current_enabled and not needs_repair,
                needs_repair=needs_repair,
                repair_reason=" ".join(repair_reasons),
                metadata={
                    "legacy_group_names": path_names,
                    "legacy_group_colors": _unique_texts(bucket["colors"]),
                    "legacy_net_key": net_key,
                },
            )
        )

    source_keys = {"schema_version", "power_pads", "current_groups", "current_enabled"}
    extra = {key: value for key, value in raw_settings.items() if key not in source_keys}
    extra.setdefault("migrated_from_schema", int(_float(raw_settings.get("schema_version", 2), 2.0)))
    return ProjectConfiguration(
        heat_sources=heat_sources,
        current_paths=paths,
        current_circuits=[],
        extra=extra,
        schema_version=SCHEMA_VERSION,
    ).normalized()


def _coerce_heat_sources(
    heat_sources: Union[ProjectConfiguration, Iterable[Union[HeatSourceDefinition, Mapping[str, Any]]]]
) -> List[HeatSourceDefinition]:
    """Normalise source input accepted by the public expansion helper."""
    if isinstance(heat_sources, ProjectConfiguration):
        return heat_sources.normalized().heat_sources
    if isinstance(heat_sources, Mapping) and "heat_sources" in heat_sources:
        return ProjectConfiguration.from_dict(heat_sources).heat_sources
    return [HeatSourceDefinition.from_dict(source) for source in (heat_sources or [])]


def _area_for_pad(pad: PadRef, pad_areas: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Find a live or persisted area for a pad without assigning identity to net data."""
    if isinstance(pad_areas, Mapping):
        candidates = (
            pad.identity_key(),
            pad.pad_uuid,
            pad.legacy_key,
            "%s-%s" % (pad.reference, pad.pad_number),
            pad.display_name,
        )
        for candidate in candidates:
            if candidate and candidate in pad_areas:
                area = _positive_or_none(pad_areas[candidate])
                if area is not None:
                    return area
    return _positive_or_none(pad.area_mm2)


def _heat_source_shares(
    source: HeatSourceDefinition, pad_areas: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Resolve a source's requested distribution and record any equal fallback."""
    source = source.normalized()
    count = len(source.pads)
    requested = source.distribution
    if requested == "custom":
        return {"shares": _normalise_shares(source.custom_shares, count), "used": "custom"}
    if requested == "equal":
        return {"shares": _normalise_shares(None, count), "used": "equal"}
    areas = [_area_for_pad(pad, pad_areas) for pad in source.pads]
    if any(area is None for area in areas):
        return {"shares": _normalise_shares(None, count), "used": "equal"}
    shares = _normalise_shares(areas, count)
    return {"shares": shares, "used": "area"}


def expand_heat_sources(
    heat_sources: Union[ProjectConfiguration, Iterable[Union[HeatSourceDefinition, Mapping[str, Any]]]],
    pad_areas: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Expand total heat-source profiles to serialisable per-pad specifications.

    Area distribution uses explicit ``pad_areas`` values first and each
    :class:`PadRef`'s saved ``area_mm2`` second.  If any pad lacks a usable
    positive area, the entire source falls back to equal shares.  Custom shares
    are normalised rather than trusted, so every expanded source preserves its
    exact total power.

    Parameters
    ----------
    heat_sources : ProjectConfiguration or iterable
        Source definitions to expand.
    pad_areas : mapping, optional
        Positive area values keyed by ``PadRef.identity_key()``, UUID, legacy
        key, ``REF-PAD`` label, or display name.

    Returns
    -------
    list of dict
        Per-pad solver-ready primitive specs.  Constant profiles contain
        ``profile.value_w``; PWL profiles contain ``profile.path`` and
        ``profile.scale``.
    """
    result: List[Dict[str, Any]] = []
    for source in _coerce_heat_sources(heat_sources):
        source = source.normalized()
        if not source.enabled:
            continue
        if source.needs_repair:
            raise ValueError(
                "Heat source '%s' needs repair%s"
                % (
                    source.name or source.id,
                    ": " + source.repair_reason if source.repair_reason else "",
                )
            )
        if not source.pads and (
            source.profile_kind == "pwl" or source.power_w != 0.0
        ):
            raise ValueError(
                "Heat source '%s' needs at least one pad"
                % (source.name or source.id)
            )
        allocation = _heat_source_shares(source, pad_areas)
        shares = allocation["shares"]
        for pad, share in zip(source.pads, shares):
            if source.profile_kind == "pwl":
                profile: Dict[str, Any] = {
                    "kind": "pwl",
                    "path": source.pwl_path,
                    "scale": share,
                }
            else:
                profile = {"kind": "constant", "value_w": source.power_w * share}
            result.append(
                {
                    "heat_source_id": source.id,
                    "heat_source_name": source.name,
                    "pad_ref": pad.to_dict(),
                    "share": share,
                    "distribution_requested": source.distribution,
                    "distribution_used": allocation["used"],
                    "profile": profile,
                }
            )
    return result


def _coerce_current_paths(
    current_paths: Union[ProjectConfiguration, Iterable[Union[CurrentPathDefinition, Mapping[str, Any]]]]
) -> List[CurrentPathDefinition]:
    """Normalise current path input accepted by the public expansion helper."""
    if isinstance(current_paths, ProjectConfiguration):
        return current_paths.normalized().current_paths
    if isinstance(current_paths, Mapping) and "current_paths" in current_paths:
        return ProjectConfiguration.from_dict(current_paths).current_paths
    return [CurrentPathDefinition.from_dict(path) for path in (current_paths or [])]


def _coerce_current_circuits(
    circuits: Optional[
        Union[ProjectConfiguration, Iterable[Union[CurrentCircuitDefinition, Mapping[str, Any]]]]
    ]
) -> List[CurrentCircuitDefinition]:
    """Normalise optional DC-circuit input accepted by current-path expansion."""
    if circuits is None:
        return []
    if isinstance(circuits, ProjectConfiguration):
        return circuits.normalized().current_circuits
    if isinstance(circuits, Mapping) and "current_circuits" in circuits:
        return ProjectConfiguration.from_dict(circuits).current_circuits
    return [CurrentCircuitDefinition.from_dict(circuit) for circuit in circuits]


def _terminal_currents(current_a: float, shares: Sequence[float], sign: float) -> List[float]:
    """Allocate a terminal magnitude with a final correction for an exact sum."""
    if not shares:
        return []
    values = [sign * current_a * share for share in shares]
    values[-1] = sign * current_a - math.fsum(values[:-1])
    return values


def expand_current_paths(
    current_paths: Union[ProjectConfiguration, Iterable[Union[CurrentPathDefinition, Mapping[str, Any]]]],
    current_circuits: Optional[
        Union[ProjectConfiguration, Iterable[Union[CurrentCircuitDefinition, Mapping[str, Any]]]]
    ] = None,
) -> List[Dict[str, Any]]:
    """Expand current paths to exactly balanced signed terminal specifications.

    Source terminals receive positive currents and sink terminals receive
    negative currents.  A linked enabled circuit overrides its paths'
    individual ``current_a`` values.  Disabled paths/circuits are omitted;
    enabled paths that need repair, reference a missing circuit, or have a
    non-zero current without both endpoint sets raise ``ValueError`` instead
    of producing an unsafe solver input.

    Parameters
    ----------
    current_paths : ProjectConfiguration or iterable
        Current path definitions to expand.
    current_circuits : ProjectConfiguration or iterable, optional
        DC circuits whose current is shared by linked paths.

    Returns
    -------
    list of dict
        JSON-primitive terminal specs with ``pad_ref``, ``role``, net display
        fields, and signed ``current_a`` suitable for resolving to existing
        ``CurrentTerminal`` objects.

    Raises
    ------
    ValueError
        If an enabled path is incomplete, needs repair, or links to an unknown
        circuit.
    """
    paths = _coerce_current_paths(current_paths)
    # A complete ProjectConfiguration is the natural controller input.  In
    # that form callers should not need to pass its circuit list a second time.
    if current_circuits is None and isinstance(current_paths, ProjectConfiguration):
        circuits = current_paths.normalized().current_circuits
    elif (
        current_circuits is None
        and isinstance(current_paths, Mapping)
        and "current_circuits" in current_paths
    ):
        circuits = ProjectConfiguration.from_dict(current_paths).current_circuits
    else:
        circuits = _coerce_current_circuits(current_circuits)
    circuit_by_id = {circuit.id: circuit.normalized() for circuit in circuits}
    result: List[Dict[str, Any]] = []

    for path in paths:
        path = path.normalized()
        if not path.enabled:
            continue
        if path.needs_repair:
            raise ValueError(
                "Current path '%s' needs repair%s"
                % (path.name or path.id, ": " + path.repair_reason if path.repair_reason else "")
            )
        current = path.current_a
        if path.circuit_id:
            circuit = circuit_by_id.get(path.circuit_id)
            if circuit is None:
                raise ValueError(
                    "Current path '%s' references missing circuit '%s'"
                    % (path.name or path.id, path.circuit_id)
                )
            if not circuit.enabled:
                continue
            if circuit.needs_repair:
                raise ValueError(
                    "Current circuit '%s' needs repair%s"
                    % (circuit.name or circuit.id, ": " + circuit.repair_reason if circuit.repair_reason else "")
                )
            current = circuit.current_a
        if current == 0.0:
            continue
        if not path.source_pads or not path.sink_pads:
            raise ValueError(
                "Current path '%s' needs at least one source and one sink pad"
                % (path.name or path.id)
            )

        source_currents = _terminal_currents(current, path.source_shares, 1.0)
        sink_currents = _terminal_currents(current, path.sink_shares, -1.0)
        # This invariant catches regressions in the final-current correction.
        if not math.isclose(math.fsum(source_currents) + math.fsum(sink_currents), 0.0, abs_tol=1e-12):
            raise ValueError("Current path '%s' could not be balanced" % (path.name or path.id))
        for role, pads, currents in (
            ("source", path.source_pads, source_currents),
            ("sink", path.sink_pads, sink_currents),
        ):
            for pad, terminal_current in zip(pads, currents):
                net_name = path.net_name or pad.net_name
                net_code = path.net_code or pad.net_code
                result.append(
                    {
                        "current_path_id": path.id,
                        "current_path_name": path.name,
                        "circuit_id": path.circuit_id,
                        "role": role,
                        "pad_ref": pad.to_dict(),
                        "name": pad.display_name
                        or "%s-%s" % (pad.reference, pad.pad_number),
                        "net_name": net_name,
                        "net_code": net_code,
                        "current_a": terminal_current,
                    }
                )
    return result


__all__ = [
    "SCHEMA_VERSION",
    "PadRef",
    "HeatSourceDefinition",
    "CurrentPathDefinition",
    "CurrentCircuitDefinition",
    "ProjectConfiguration",
    "load_project_config",
    "save_project_config",
    "migrate_v2_settings",
    "expand_heat_sources",
    "expand_current_paths",
]
