"""
Minimal wx mock for testing without wxPython.

This module provides stub implementations of wx classes
used by ThermalSim.
"""

import sys


class _WxAdvMock:
    """Mock wx.adv module."""

    class HyperlinkCtrl:
        def __init__(self, parent, id=-1, label="", url="", **kwargs):
            self.label = label
            self.url = url

        def SetToolTip(self, tip):
            pass


class _WxMock:
    """Mock wx module."""

    ID_OK = 5100
    ID_CANCEL = 5101
    ID_ANY = -1
    DD_DEFAULT_STYLE = 0
    FD_OPEN = 0x01
    FD_FILE_MUST_EXIST = 0x02
    SP_ARROW_KEYS = 0x1000
    SP_WRAP = 0x2000

    adv = _WxAdvMock()

    class Dialog:
        def __init__(self, *args, **kwargs):
            pass

        def ShowModal(self):
            return _WxMock.ID_OK

        def Destroy(self):
            pass

        def SetSizer(self, sizer):
            pass

        def Fit(self):
            pass

        def Center(self):
            pass

        def SetSize(self, size):
            pass

        def SetMinSize(self, size):
            pass

        def SetToolTip(self, tip):
            pass

    class Panel:
        def __init__(self, parent, **kwargs):
            pass

        def SetSizer(self, sizer):
            pass

        def SetToolTip(self, tip):
            pass

    class Notebook:
        def __init__(self, parent, **kwargs):
            pass

        def AddPage(self, page, caption):
            pass

        def SetToolTip(self, tip):
            pass

    class BoxSizer:
        VERTICAL = 1
        HORIZONTAL = 0

        def __init__(self, orient=0):
            self.orient = orient

        def Add(self, *args, **kwargs):
            pass

        def AddStretchSpacer(self):
            pass

    class StaticBoxSizer(BoxSizer):
        def __init__(self, orient, parent, label=""):
            super().__init__(orient)

    class StaticText:
        def __init__(self, parent, label="", size=None, **kwargs):
            self.label = label

        def SetToolTip(self, tip):
            pass

        def SetLabel(self, label):
            self.label = label

        def GetFont(self):
            return _WxMock.Font()

        def SetFont(self, font):
            pass

        def SetForegroundColour(self, colour):
            pass

    class StaticLine:
        def __init__(self, parent, **kwargs):
            pass

        def SetToolTip(self, tip):
            pass

    class TextCtrl:
        def __init__(self, parent, value="", style=0, **kwargs):
            self._value = value

        def GetValue(self):
            return self._value

        def SetValue(self, value):
            self._value = str(value)

        def AppendText(self, text):
            self._value += str(text)

        def Enable(self, enable=True):
            pass

        def SetMinSize(self, size):
            pass

        def SetToolTip(self, tip):
            pass

    class SpinCtrlDouble:
        def __init__(self, parent, value="", min=0.0, max=100.0, inc=1.0,
                     style=0, **kwargs):
            self._value = float(value) if value else min
            self._min = min
            self._max = max

        def GetValue(self):
            return self._value

        def SetValue(self, value):
            self._value = float(value)

        def SetDigits(self, digits):
            pass

        def Enable(self, enable=True):
            pass

        def SetToolTip(self, tip):
            pass

        def SetMinSize(self, size):
            pass

    class SpinCtrl:
        def __init__(self, parent, value="", min=0, max=100,
                     style=0, **kwargs):
            self._value = int(value) if value else min
            self._min = min
            self._max = max

        def GetValue(self):
            return self._value

        def SetValue(self, value):
            self._value = int(value)

        def Enable(self, enable=True):
            pass

        def SetToolTip(self, tip):
            pass

        def SetMinSize(self, size):
            pass

    class CheckBox:
        def __init__(self, parent, label=""):
            self._value = False

        def GetValue(self):
            return self._value

        def SetValue(self, value):
            self._value = bool(value)

        def Bind(self, event, handler):
            pass

        def SetToolTip(self, tip):
            pass

    class Button:
        def __init__(self, parent, id=None, label=""):
            self.label = label
            self._enabled = True

        def Bind(self, event, handler):
            pass

        def SetToolTip(self, tip):
            pass

        def Enable(self, enable=True):
            self._enabled = enable

        def SetLabel(self, label):
            self.label = label

    class Choice:
        def __init__(self, parent, choices=None, **kwargs):
            self._items = list(choices) if choices else []
            self._selection = -1
            self._data = []

        def SetItems(self, items):
            self._items = list(items)
            self._selection = -1
            self._data = []

        def Clear(self):
            self._items = []
            self._selection = -1
            self._data = []

        def Append(self, item, clientData=None):
            self._items.append(item)
            self._data.append(clientData)
            return len(self._items) - 1

        def GetSelection(self):
            return self._selection

        def SetSelection(self, n):
            self._selection = n

        def GetString(self, n):
            if 0 <= n < len(self._items):
                return self._items[n]
            return ""

        def GetClientData(self, n):
            if 0 <= n < len(self._data):
                return self._data[n]
            return None

        def GetCount(self):
            return len(self._items)

        def Bind(self, event, handler):
            pass

        def SetToolTip(self, tip):
            pass

        def Enable(self, enable=True):
            pass

    class ListCtrl:
        def __init__(self, parent, style=0, **kwargs):
            self._columns = []
            self._items = []

        def InsertColumn(self, col, heading, width=80):
            self._columns.append(heading)

        def InsertItem(self, index, label):
            while len(self._items) <= index:
                self._items.append({})
            self._items[index] = {0: label}
            return index

        def SetItem(self, index, col, label):
            if index < len(self._items):
                self._items[index][col] = label

        def DeleteAllItems(self):
            self._items = []

        def GetFirstSelected(self):
            return -1

        def SetMinSize(self, size):
            pass

        def SetToolTip(self, tip):
            pass

    class Colour:
        def __init__(self, r=0, g=0, b=0, a=255):
            self.r, self.g, self.b, self.a = r, g, b, a

    class DirDialog:
        def __init__(self, parent, message="", defaultPath="", style=0):
            self.path = defaultPath

        def ShowModal(self):
            return _WxMock.ID_OK

        def GetPath(self):
            return self.path

        def Destroy(self):
            pass

    class FileDialog:
        def __init__(self, parent, message="", defaultDir="", wildcard="",
                     style=0, **kwargs):
            self._path = ""

        def ShowModal(self):
            return _WxMock.ID_CANCEL

        def GetPath(self):
            return self._path

        def Destroy(self):
            pass

    # Constants
    VERTICAL = 1
    HORIZONTAL = 0
    ALL = 0x0F
    EXPAND = 0x01
    RIGHT = 0x02
    LEFT = 0x04
    TOP = 0x08
    BOTTOM = 0x10
    TE_MULTILINE = 0x01
    TE_READONLY = 0x02
    TE_DONTWRAP = 0x04
    ALIGN_CENTER_VERTICAL = 0x08
    ALIGN_RIGHT = 0x10
    LI_HORIZONTAL = 0
    LC_REPORT = 0x01
    LC_SINGLE_SEL = 0x02
    FONTWEIGHT_BOLD = 92
    PD_CAN_ABORT = 0x01
    PD_APP_MODAL = 0x02
    PD_REMAINING_TIME = 0x04
    PD_AUTO_HIDE = 0x08

    # Event types
    EVT_BUTTON = "EVT_BUTTON"
    EVT_CHECKBOX = "EVT_CHECKBOX"
    EVT_CHOICE = "EVT_CHOICE"

    @staticmethod
    def CallAfter(func, *args, **kwargs):
        """Mock wx.CallAfter — execute immediately in test context."""
        func(*args, **kwargs)

    @staticmethod
    def MessageBox(message, caption="", style=0):
        """Mock wx.MessageBox."""
        pass

    class Font:
        """Mock wx.Font."""
        def SetWeight(self, weight):
            pass
        def GetWeight(self):
            return 0


def install_wx_mock():
    """Install wx mock into sys.modules."""
    mock = _WxMock()
    sys.modules['wx'] = mock
    sys.modules['wx.adv'] = _WxAdvMock()


def uninstall_wx_mock():
    """Remove wx mock from sys.modules."""
    if 'wx' in sys.modules and isinstance(sys.modules['wx'], type(_WxMock)):
        del sys.modules['wx']
    if 'wx.adv' in sys.modules:
        del sys.modules['wx.adv']
