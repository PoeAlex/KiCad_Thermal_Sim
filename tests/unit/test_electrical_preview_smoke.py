import math
import os

import numpy as np

from mocks.pcbnew_mock import EDA_RECT, F_Cu, MockBoard, MockFootprint, MockPad, MockZone, PAD_ATTRIB_SMD, VECTOR2I
from ThermalSim.electrical_solver import CurrentTerminal, ElectricalConfig, prepare_electrical_geometry
from ThermalSim.visualization import save_electrical_connectivity_preview


def iu(mm):
    return int(round(mm * 1e6))


class RingZone(MockZone):
    def __init__(self):
        super().__init__(
            layers=[F_Cu],
            bbox=EDA_RECT(iu(1), iu(1), iu(8), iu(8)),
            filled=True,
            net_code=21,
            net_name="PREVIEW_ZONE",
        )

    def HitTestFilledArea(self, layer_id, pos, margin=0):
        if layer_id != F_Cu:
            return False
        x, y = pos.x * 1e-6, pos.y * 1e-6
        r = math.hypot(x - 5.0, y - 5.0)
        return 2.0 <= r <= 4.0


def test_preview_writes_topology_graph_and_occupancy_images(tmp_path):
    cfg = ElectricalConfig(
        copper_ids=[F_Cu], rows=100, cols=100,
        x_min=0.0, y_min=0.0, res=0.1,
        t_cu=np.array([35e-6]),
        layer_names={F_Cu: "F.Cu"},
    )
    pad = MockPad(
        position=VECTOR2I(iu(5.0), iu(9.2)), layer=F_Cu,
        attribute=PAD_ATTRIB_SMD,
        bbox=EDA_RECT(iu(4.6), iu(8.8), iu(0.8), iu(0.8)),
        net_code=21, net_name="PREVIEW_ZONE", number="1", layers=[F_Cu],
    )
    board = MockBoard(footprints=[MockFootprint("J1", [pad])], zones=[RingZone()], layer_names={F_Cu: "F.Cu"})
    terminal = CurrentTerminal(pad, "J1-1", "PREVIEW_ZONE", 21, 1.0)
    prepared = prepare_electrical_geometry(board, [terminal], cfg)

    path = save_electrical_connectivity_preview(
        prepared, cfg, [terminal], ["F.Cu"],
        out_dir=str(tmp_path), open_file=False,
    )

    assert path == os.path.join(str(tmp_path), "electrical_connectivity_preview.png")
    for name in (
        "electrical_connectivity_preview.png",
        "electrical_graph_raster_preview.png",
        "electrical_copper_occupancy_preview.png",
    ):
        output = tmp_path / name
        assert output.exists()
        assert output.stat().st_size > 1000
