"""
KiCad pcbnew mock objects for testing.

This module provides mock implementations of KiCad's pcbnew API
to allow testing without a running KiCad instance.

Classes
-------
VECTOR2I
    Mock 2D vector with integer coordinates.
EDA_RECT
    Mock bounding box rectangle.
MockPad
    Mock PCB pad object.
MockTrack
    Mock PCB track (trace) object.
MockVia
    Mock PCB via object.
MockZone
    Mock copper zone object.
MockFootprint
    Mock component footprint.
MockBoard
    Mock PCB board object.
"""

import sys
from typing import List, Optional, Set


# Layer ID constants
F_Cu = 0
In1_Cu = 1
In2_Cu = 2
In3_Cu = 3
In4_Cu = 4
B_Cu = 31
Eco1_User = 46

# Pad attribute constants
PAD_ATTRIB_PTH = 0
PAD_ATTRIB_SMD = 1
PAD_ATTRIB_CONN = 2
PAD_ATTRIB_NPTH = 3


class VECTOR2I:
    """
    Mock 2D vector with integer coordinates.

    Parameters
    ----------
    x : int
        X coordinate in nm (internal units).
    y : int
        Y coordinate in nm (internal units).
    """

    def __init__(self, x: int = 0, y: int = 0):
        self.x = int(x)
        self.y = int(y)

    def __repr__(self):
        return f"VECTOR2I({self.x}, {self.y})"

    def __eq__(self, other):
        if isinstance(other, VECTOR2I):
            return self.x == other.x and self.y == other.y
        return False


class EDA_RECT:
    """
    Mock bounding box rectangle.

    Parameters
    ----------
    x : int
        Left edge X coordinate in nm.
    y : int
        Top edge Y coordinate in nm.
    width : int
        Width in nm.
    height : int
        Height in nm.
    """

    def __init__(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0):
        self._x = int(x)
        self._y = int(y)
        self._width = int(width)
        self._height = int(height)

    def GetX(self) -> int:
        return self._x

    def GetY(self) -> int:
        return self._y

    def GetWidth(self) -> int:
        return self._width

    def GetHeight(self) -> int:
        return self._height

    def SetX(self, x: int):
        self._x = int(x)

    def SetY(self, y: int):
        self._y = int(y)

    def SetWidth(self, w: int):
        self._width = int(w)

    def SetHeight(self, h: int):
        self._height = int(h)

    def Contains(self, point: VECTOR2I) -> bool:
        return (self._x <= point.x <= self._x + self._width and
                self._y <= point.y <= self._y + self._height)

    def __repr__(self):
        return f"EDA_RECT({self._x}, {self._y}, {self._width}, {self._height})"


class MockLayerSet:
    """Mock layer set for zones spanning multiple layers."""

    def __init__(self, layers: Optional[List[int]] = None):
        self._layers = set(layers) if layers else set()

    def Contains(self, layer_id: int) -> bool:
        return layer_id in self._layers

    def IntSeq(self) -> List[int]:
        return list(self._layers)

    def AddLayer(self, layer_id: int):
        self._layers.add(layer_id)


class MockPad:
    """
    Mock PCB pad object.

    Parameters
    ----------
    position : VECTOR2I, optional
        Pad center position in nm.
    layer : int, optional
        Pad layer ID.
    attribute : int, optional
        Pad attribute (PTH, SMD, etc.).
    bbox : EDA_RECT, optional
        Bounding box.
    selected : bool, optional
        Whether pad is selected.
    net_code : int, optional
        Net code for connectivity.
    net_name : str, optional
        Net name.
    number : str, optional
        Pad number/name.
    """

    def __init__(
        self,
        position: Optional[VECTOR2I] = None,
        layer: int = F_Cu,
        attribute: int = PAD_ATTRIB_SMD,
        bbox: Optional[EDA_RECT] = None,
        selected: bool = False,
        net_code: int = 0,
        net_name: str = "",
        number: str = "1"
    ):
        self._position = position or VECTOR2I(0, 0)
        self._layer = layer
        self._attribute = attribute
        self._bbox = bbox or EDA_RECT(
            self._position.x - 500000,
            self._position.y - 500000,
            1000000,
            1000000
        )
        self._selected = selected
        self._net_code = net_code
        self._net_name = net_name
        self._number = number

    def GetPosition(self) -> VECTOR2I:
        return self._position

    def GetLayer(self) -> int:
        return self._layer

    def GetAttribute(self) -> int:
        return self._attribute

    def GetBoundingBox(self) -> EDA_RECT:
        return self._bbox

    def IsSelected(self) -> bool:
        return self._selected

    def GetNetCode(self) -> int:
        return self._net_code

    def GetNetname(self) -> str:
        return self._net_name

    def GetNumber(self) -> str:
        return self._number


class MockNet:
    """Mock net object."""

    def __init__(self, name: str = "", code: int = 0):
        self._name = name
        self._code = code

    def GetNetname(self) -> str:
        return self._name

    def GetNetCode(self) -> int:
        return self._code


class MockTrack:
    """
    Mock PCB track (trace) object.

    Parameters
    ----------
    layer : int, optional
        Track layer ID.
    bbox : EDA_RECT, optional
        Bounding box.
    start : VECTOR2I, optional
        Start point in internal units (nm).
    end : VECTOR2I, optional
        End point in internal units (nm).
    width : int, optional
        Track width in internal units (nm).
    net_code : int, optional
        Net code for connectivity.
    """

    def __init__(self, layer: int = F_Cu, bbox: Optional[EDA_RECT] = None,
                 start: Optional[VECTOR2I] = None, end: Optional[VECTOR2I] = None,
                 width: int = 250000, net_code: int = 0):
        self._layer = layer
        self._start = start or VECTOR2I(0, 0)
        self._end = end or VECTOR2I(1000000, 0)
        self._width = width
        self._net_code = net_code
        if bbox is not None:
            self._bbox = bbox
        else:
            # Auto-compute bbox from start/end/width
            x0 = min(self._start.x, self._end.x) - width // 2
            y0 = min(self._start.y, self._end.y) - width // 2
            x1 = max(self._start.x, self._end.x) + width // 2
            y1 = max(self._start.y, self._end.y) + width // 2
            self._bbox = EDA_RECT(x0, y0, x1 - x0, y1 - y0)

    def GetLayer(self) -> int:
        return self._layer

    def GetBoundingBox(self) -> EDA_RECT:
        return self._bbox

    def GetStart(self) -> VECTOR2I:
        return self._start

    def GetEnd(self) -> VECTOR2I:
        return self._end

    def GetWidth(self) -> int:
        return self._width

    def GetNetCode(self) -> int:
        return self._net_code


class MockVia(MockTrack):
    """
    Mock PCB via object.

    Inherits from MockTrack but with type name containing 'VIA'.
    """

    def __init__(self, bbox: Optional[EDA_RECT] = None, layers: Optional[List[int]] = None,
                 net_code: int = 0, position: Optional[VECTOR2I] = None):
        pos = position or VECTOR2I(0, 0)
        super().__init__(layer=F_Cu, bbox=bbox, start=pos, end=pos, net_code=net_code)
        self._layers = layers or [F_Cu, B_Cu]

    def __class__(self):
        class _VIA:
            pass
        return _VIA


class MockZone:
    """
    Mock copper zone object.

    Parameters
    ----------
    layers : list of int, optional
        Layer IDs this zone exists on.
    bbox : EDA_RECT, optional
        Bounding box.
    filled : bool, optional
        Whether zone is filled.
    net_code : int, optional
        Net code.
    net_name : str, optional
        Net name.
    is_rule_area : bool, optional
        Whether this is a rule area (keepout).
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        bbox: Optional[EDA_RECT] = None,
        filled: bool = True,
        net_code: int = 0,
        net_name: str = "GND",
        is_rule_area: bool = False,
        fill_mask: Optional[Set[tuple]] = None,
        hit_test_func=None
    ):
        self._layers = layers or [F_Cu]
        self._layer_set = MockLayerSet(self._layers)
        self._bbox = bbox or EDA_RECT(0, 0, 10000000, 10000000)
        self._filled = filled
        self._net_code = net_code
        self._net_name = net_name
        self._is_rule_area = is_rule_area
        # fill_mask: set of (x_nm, y_nm) positions that are filled
        self._fill_mask = fill_mask
        self._hit_test_func = hit_test_func

    def GetLayer(self) -> int:
        return self._layers[0] if self._layers else F_Cu

    def GetLayerSet(self) -> MockLayerSet:
        return self._layer_set

    def GetBoundingBox(self) -> EDA_RECT:
        return self._bbox

    def IsFilled(self) -> bool:
        return self._filled

    def IsOnLayer(self, layer_id: int) -> bool:
        return layer_id in self._layers

    def GetNetCode(self) -> int:
        return self._net_code

    def GetNetname(self) -> str:
        return self._net_name

    def GetNet(self) -> MockNet:
        return MockNet(self._net_name, self._net_code)

    def GetIsRuleArea(self) -> bool:
        return self._is_rule_area

    def GetIsKeepout(self) -> bool:
        return self._is_rule_area

    def HitTestFilledArea(self, layer_id: int, pos: VECTOR2I, margin: int = 0) -> bool:
        """Test if position is within filled area of zone."""
        if not self._filled:
            return False
        if layer_id not in self._layers:
            return False
        if self._fill_mask is not None:
            # Check fill mask with some tolerance
            for fx, fy in self._fill_mask:
                if abs(pos.x - fx) <= margin + 100000 and abs(pos.y - fy) <= margin + 100000:
                    return True
            return False
        # Use custom hit test function if provided
        if self._hit_test_func is not None:
            return self._hit_test_func(pos)
        # Default: check bounding box
        return self._bbox.Contains(pos)

    def HitTest(self, pos: VECTOR2I) -> bool:
        if self._hit_test_func is not None:
            return self._hit_test_func(pos)
        return self._bbox.Contains(pos)


class MockDrawing:
    """Mock drawing object (shapes on User.Eco1, etc.)."""

    def __init__(self, layer: int = Eco1_User, bbox: Optional[EDA_RECT] = None,
                 hit_test_func=None, filled: bool = False):
        self._layer = layer
        self._bbox = bbox or EDA_RECT(0, 0, 5000000, 5000000)
        self._hit_test_func = hit_test_func
        self._filled = filled

    def GetLayer(self) -> int:
        return self._layer

    def GetBoundingBox(self) -> EDA_RECT:
        return self._bbox

    def IsFilled(self) -> bool:
        return self._filled

    def HitTest(self, pos: VECTOR2I) -> bool:
        if self._hit_test_func is not None:
            return self._hit_test_func(pos)
        return self._bbox.Contains(pos)


class MockFootprint:
    """
    Mock component footprint.

    Parameters
    ----------
    reference : str, optional
        Reference designator (e.g., "U1").
    pads : list of MockPad, optional
        Pads belonging to this footprint.
    graphical_items : list, optional
        Graphical items (drawings) embedded in this footprint.
    """

    def __init__(self, reference: str = "U1", pads: Optional[List[MockPad]] = None,
                 graphical_items: Optional[List] = None):
        self._reference = reference
        self._pads = pads or []
        self._graphical_items = graphical_items or []

    def GetReference(self) -> str:
        return self._reference

    def Pads(self) -> List[MockPad]:
        return self._pads

    def GraphicalItems(self) -> List:
        return self._graphical_items


class MockBoard:
    """
    Mock PCB board object.

    Parameters
    ----------
    filename : str, optional
        Board filename.
    footprints : list of MockFootprint, optional
        Footprints on the board.
    tracks : list, optional
        Tracks and vias on the board.
    zones : list of MockZone, optional
        Copper zones on the board.
    drawings : list of MockDrawing, optional
        Drawing objects.
    layer_names : dict, optional
        Mapping of layer ID to name.
    """

    def __init__(
        self,
        filename: str = "",
        footprints: Optional[List[MockFootprint]] = None,
        tracks: Optional[List] = None,
        zones: Optional[List[MockZone]] = None,
        drawings: Optional[List[MockDrawing]] = None,
        layer_names: Optional[dict] = None
    ):
        self._filename = filename
        self._footprints = footprints or []
        self._tracks = tracks or []
        self._zones = zones or []
        self._drawings = drawings or []
        self._layer_names = layer_names or {
            F_Cu: "F.Cu",
            In1_Cu: "In1.Cu",
            In2_Cu: "In2.Cu",
            B_Cu: "B.Cu",
            Eco1_User: "User.Eco1"
        }

    def GetFileName(self) -> str:
        return self._filename

    def Footprints(self) -> List[MockFootprint]:
        return self._footprints

    def GetFootprints(self) -> List[MockFootprint]:
        return self._footprints

    def Tracks(self) -> List:
        return self._tracks

    def GetTracks(self) -> List:
        return self._tracks

    def Zones(self) -> List[MockZone]:
        return self._zones

    def GetZones(self) -> List[MockZone]:
        return self._zones

    def GetDrawings(self) -> List[MockDrawing]:
        return self._drawings

    def GetLayerName(self, layer_id: int) -> str:
        return self._layer_names.get(layer_id, f"Layer_{layer_id}")

    def GetLayerID(self, name: str) -> int:
        for lid, lname in self._layer_names.items():
            if lname == name:
                return lid
        return -1


class MockZoneFiller:
    """Mock zone filler."""

    def __init__(self, board: MockBoard):
        self._board = board

    def Fill(self, zones):
        pass


class MockActionPlugin:
    """Mock ActionPlugin base class for KiCad plugins."""

    def __init__(self):
        self.name = ""
        self.category = ""
        self.description = ""
        self.show_toolbar_button = False
        self.icon_file_name = ""

    def defaults(self):
        pass

    def Run(self):
        pass

    def register(self):
        pass


# Utility functions
def FromMM(mm: float) -> int:
    """Convert mm to internal units (nm)."""
    return int(mm * 1e6)


def ToMM(nm: int) -> float:
    """Convert internal units (nm) to mm."""
    return nm * 1e-6


def GetBoard() -> Optional[MockBoard]:
    """Get the active board (returns None in mock)."""
    return None


# Module-level mock module for sys.modules injection
class _MockPcbnewModule:
    """Mock pcbnew module for sys.modules injection."""

    VECTOR2I = VECTOR2I
    EDA_RECT = EDA_RECT

    PAD_ATTRIB_PTH = PAD_ATTRIB_PTH
    PAD_ATTRIB_SMD = PAD_ATTRIB_SMD
    PAD_ATTRIB_CONN = PAD_ATTRIB_CONN
    PAD_ATTRIB_NPTH = PAD_ATTRIB_NPTH

    F_Cu = F_Cu
    In1_Cu = In1_Cu
    In2_Cu = In2_Cu
    In3_Cu = In3_Cu
    In4_Cu = In4_Cu
    B_Cu = B_Cu
    Eco1_User = Eco1_User

    FromMM = staticmethod(FromMM)
    ToMM = staticmethod(ToMM)
    GetBoard = staticmethod(GetBoard)

    ZONE_FILLER = MockZoneFiller
    ActionPlugin = MockActionPlugin


_original_pcbnew = None


def install_mock():
    """
    Install the mock pcbnew module into sys.modules.

    This allows code that imports pcbnew to use the mock instead.
    Call uninstall_mock() to restore the original module.
    """
    global _original_pcbnew
    _original_pcbnew = sys.modules.get('pcbnew')
    sys.modules['pcbnew'] = _MockPcbnewModule()


def uninstall_mock():
    """
    Restore the original pcbnew module.
    """
    global _original_pcbnew
    if _original_pcbnew is not None:
        sys.modules['pcbnew'] = _original_pcbnew
    elif 'pcbnew' in sys.modules:
        del sys.modules['pcbnew']
    _original_pcbnew = None
