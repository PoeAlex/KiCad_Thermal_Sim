import math

import numpy as np

from mocks.pcbnew_mock import (
    EDA_RECT,
    F_Cu,
    In1_Cu,
    MockBoard,
    MockFootprint,
    MockPad,
    MockTrack,
    MockVia,
    MockZone,
    PAD_ATTRIB_SMD,
    VECTOR2I,
)

from ThermalSim.electrical_solver import (
    CurrentTerminal,
    ElectricalConfig,
    _track_contact_with_kind,
    net_key_from_values,
    prepare_electrical_geometry,
    solve_electrical_heating,
)


def iu(mm):
    return int(round(mm * 1e6))


def rect_mm(x, y, w, h):
    return EDA_RECT(iu(x), iu(y), iu(w), iu(h))


def point_mm(x, y):
    return VECTOR2I(iu(x), iu(y))


def track_bbox(start, end, width):
    sx, sy = start
    ex, ey = end
    r = 0.5 * width
    x0 = min(sx, ex) - r
    y0 = min(sy, ey) - r
    x1 = max(sx, ex) + r
    y1 = max(sy, ey) + r
    return rect_mm(x0, y0, x1 - x0, y1 - y0)


def make_track(start, end, width, net_code=1, net_name="DEMO", layer=F_Cu):
    return MockTrack(
        layer=layer,
        bbox=track_bbox(start, end, width),
        start=point_mm(*start),
        end=point_mm(*end),
        width=iu(width),
        net_code=net_code,
        net_name=net_name,
    )


def make_pad(x, y, size=0.8, net_code=1, net_name="DEMO", layer=F_Cu, number="1"):
    return MockPad(
        position=point_mm(x, y),
        layer=layer,
        attribute=PAD_ATTRIB_SMD,
        bbox=rect_mm(x - size / 2, y - size / 2, size, size),
        net_code=net_code,
        net_name=net_name,
        number=number,
        layers=[layer],
    )


def config(x_min=0.0, y_min=0.0, width=10.0, height=10.0, res=0.1, layers=(F_Cu,)):
    return ElectricalConfig(
        copper_ids=list(layers),
        rows=int(math.ceil(height / res)),
        cols=int(math.ceil(width / res)),
        x_min=x_min,
        y_min=y_min,
        res=res,
        t_cu=np.full(len(layers), 35e-6, dtype=np.float64),
        layer_names={F_Cu: "F.Cu", In1_Cu: "In1.Cu"},
    )


class AnnularZone(MockZone):
    """Filled annulus used to reproduce the historical bbox-probe failure."""

    def __init__(self, cx, cy, r_outer, r_inner, *, layer=F_Cu, net_code=1, net_name="DEMO"):
        self.cx = float(cx)
        self.cy = float(cy)
        self.r_outer = float(r_outer)
        self.r_inner = float(r_inner)
        super().__init__(
            layers=[layer],
            bbox=rect_mm(cx - r_outer, cy - r_outer, 2 * r_outer, 2 * r_outer),
            filled=True,
            net_code=net_code,
            net_name=net_name,
        )

    def HitTestFilledArea(self, layer_id, pos, margin=0):
        if layer_id not in self._layers:
            return False
        x = pos.x * 1e-6
        y = pos.y * 1e-6
        r = math.hypot(x - self.cx, y - self.cy)
        margin_mm = max(0.0, float(margin) * 1e-6)
        return (self.r_inner - margin_mm) <= r <= (self.r_outer + margin_mm)


class LayerSelectiveZone(MockZone):
    """Same KiCad zone object with different filled copper per layer."""

    def __init__(self, *, net_code=1, net_name="DEMO"):
        super().__init__(
            layers=[F_Cu, In1_Cu],
            bbox=rect_mm(2.0, 2.0, 6.0, 6.0),
            filled=True,
            net_code=net_code,
            net_name=net_name,
        )

    def HitTestFilledArea(self, layer_id, pos, margin=0):
        x = pos.x * 1e-6
        y = pos.y * 1e-6
        if layer_id == F_Cu:
            return 2.0 <= x <= 4.0 and 2.0 <= y <= 4.0
        if layer_id == In1_Cu:
            return 5.0 <= x <= 8.0 and 5.0 <= y <= 8.0
        return False


def fake_terminal(net_code=1, net_name="DEMO"):
    # prepare_electrical_geometry only needs the net identity from this object.
    return CurrentTerminal(None, "selector", net_name, net_code, 1.0)


def contact_kinds(prepared, net_code=1, net_name="DEMO"):
    key = net_key_from_values(net_code, net_name)
    return [item[5] for item in prepared.contact_points.get(key, [])]


def test_parallel_025mm_same_net_tracks_remain_separate():
    """Regression: close same-net winding turns must not short sidewall-to-sidewall."""
    a = make_track((1.0, 2.0), (9.0, 2.0), 0.25)
    b = make_track((1.0, 2.45), (9.0, 2.45), 0.25)  # 0.20 mm copper gap

    contact, kind = _track_contact_with_kind(a, b)
    assert contact is None
    assert kind is None

    board = MockBoard(tracks=[a, b])
    prepared = prepare_electrical_geometry(board, [fake_terminal()], config(res=0.4))
    assert contact_kinds(prepared) == []


def test_15um_endpoint_gap_snaps_but_50um_gap_does_not_for_025mm_tracks():
    """Regression: generated winding endpoint gaps snap only inside the 25 um tolerance."""
    left = make_track((1.0, 2.0), (4.0, 2.0), 0.25)
    near = make_track((4.015, 2.0), (7.0, 2.0), 0.25)
    far = make_track((4.050, 3.0), (7.0, 3.0), 0.25)
    far_left = make_track((1.0, 3.0), (4.0, 3.0), 0.25)

    point, kind = _track_contact_with_kind(left, near)
    assert point is not None
    assert kind == "track_endpoint_snap"

    point, kind = _track_contact_with_kind(far_left, far)
    assert point is None
    assert kind is None


def test_track_to_annular_zone_connects_from_raster_overlap_even_when_endpoint_is_in_hole():
    """Regression for the old bbox endpoint probe: endpoint is in the zone hole."""
    net_name = "ZONE_PATH"
    net_code = 11
    zone = AnnularZone(5.0, 5.0, 3.0, 2.0, net_code=net_code, net_name=net_name)
    left = make_track((0.75, 5.0), (5.0, 5.0), 0.40, net_code, net_name)
    right = make_track((9.25, 5.0), (5.0, 5.0), 0.40, net_code, net_name)

    # The historical single-point bbox test sampled the endpoint at the centre.
    assert zone.GetBoundingBox().Contains(left.GetEnd())
    assert not zone.HitTestFilledArea(F_Cu, left.GetEnd(), 1)

    board = MockBoard(tracks=[left, right], zones=[zone])
    prepared = prepare_electrical_geometry(
        board,
        [fake_terminal(net_code, net_name)],
        config(res=0.10),
    )
    kinds = contact_kinds(prepared, net_code, net_name)
    assert kinds.count("track_to_shape_raster_overlap") == 2


def test_zone_via_contact_uses_actual_raster_overlap_when_bbox_midpoint_is_in_zone_hole():
    """Regression: a via can overlap zone copper although the bbox midpoint lies in a void."""
    net_name = "ZONE_VIA"
    net_code = 12
    zone = AnnularZone(5.0, 5.0, 1.5, 0.6, net_code=net_code, net_name=net_name)
    # Via centre is inside the annular hole, but its 1.2 mm copper bbox reaches the annulus.
    via = MockVia(
        bbox=rect_mm(4.95, 4.40, 1.20, 1.20),
        layers=[F_Cu],
        net_code=net_code,
        net_name=net_name,
    )

    midpoint = point_mm(5.55, 5.0)
    assert not zone.HitTestFilledArea(F_Cu, midpoint, 1)

    board = MockBoard(tracks=[via], zones=[zone])
    prepared = prepare_electrical_geometry(
        board,
        [fake_terminal(net_code, net_name)],
        config(res=0.05),
    )
    assert "shape_raster_overlap" in contact_kinds(prepared, net_code, net_name)


def test_multilayer_zone_rasterization_uses_the_requested_copper_layer():
    """Regression: each per-layer zone primitive must use that layer's fill, not zone.GetLayer()."""
    net_name = "MULTILAYER_ZONE"
    net_code = 13
    zone = LayerSelectiveZone(net_code=net_code, net_name=net_name)
    track = make_track(
        (5.5, 6.5), (7.5, 6.5), 0.40,
        net_code, net_name, layer=In1_Cu,
    )
    board = MockBoard(
        tracks=[track],
        zones=[zone],
        layer_names={F_Cu: "F.Cu", In1_Cu: "In1.Cu"},
    )
    prepared = prepare_electrical_geometry(
        board,
        [fake_terminal(net_code, net_name)],
        config(res=0.10, layers=(F_Cu, In1_Cu)),
    )
    assert "track_to_shape_raster_overlap" in contact_kinds(prepared, net_code, net_name)


def test_balanced_terminal_path_through_annular_zone_solves_without_island_error():
    """Integration regression: balanced pads joined only by track/zone raster contacts must solve."""
    net_name = "ZONE_SOLVE"
    net_code = 14
    src = make_pad(0.75, 5.0, net_code=net_code, net_name=net_name, number="1")
    sink = make_pad(9.25, 5.0, net_code=net_code, net_name=net_name, number="1")
    zone = AnnularZone(5.0, 5.0, 3.0, 2.0, net_code=net_code, net_name=net_name)
    left = make_track((0.75, 5.0), (5.0, 5.0), 0.40, net_code, net_name)
    right = make_track((9.25, 5.0), (5.0, 5.0), 0.40, net_code, net_name)
    board = MockBoard(
        footprints=[MockFootprint("J1", [src]), MockFootprint("J2", [sink])],
        tracks=[left, right],
        zones=[zone],
        layer_names={F_Cu: "F.Cu"},
    )
    terms = [
        CurrentTerminal(src, "J1-1", net_name, net_code, +0.5),
        CurrentTerminal(sink, "J2-1", net_name, net_code, -0.5),
    ]

    result = solve_electrical_heating(board, terms, config(res=0.10))
    assert result.valid, result.errors
    assert not result.errors
    assert result.net_summaries
    assert result.net_summaries[0].contact_edge_count >= 4
    assert result.net_summaries[0].effective_resistance_ohm is not None
