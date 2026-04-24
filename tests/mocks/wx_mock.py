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
    LC_REPORT = 0x4000
    LC_SINGLE_SEL = 0x8000
    LIST_NEXT_ALL = 0
    LIST_STATE_SELECTED = 1

    adv = _WxAdvMock()

    class Dialog:
        def __init__(self, *args, **kwargs):
            pass

        def ShowModal(self):
            return _WxMock.ID_OK

        def Show(self, show=True):
            return True

        def Raise(self):
            pass

        def Destroy(self):
            pass

        def EndModal(self, code):
            self._modal_result = code

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
            self._pages = []

        def AddPage(self, page, caption):
            self._pages.append((page, caption))

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

        def SetMinSize(self, size):
            pass

        def Bind(self, event, handler):
            pass

    class Choice:
        def __init__(self, parent, choices=None, **kwargs):
            self._choices = list(choices or [])
            self._selection = 0 if self._choices else -1

        def SetSelection(self, selection):
            self._selection = int(selection)

        def GetSelection(self):
            return self._selection

        def GetStringSelection(self):
            if 0 <= self._selection < len(self._choices):
                return self._choices[self._selection]
            return ""

        def SetToolTip(self, tip):
            pass

        def Bind(self, event, handler):
            pass

    class ListCtrl:
        def __init__(self, parent, style=0, **kwargs):
            self._columns = []
            self._items = []
            self._selected = -1

        def InsertColumn(self, idx, heading, width=-1):
            if idx >= len(self._columns):
                self._columns.extend([""] * (idx - len(self._columns) + 1))
            self._columns[idx] = heading

        def DeleteAllItems(self):
            self._items = []
            self._selected = -1

        def InsertItem(self, idx, text):
            row = ["" for _ in range(max(1, len(self._columns)))]
            row[0] = str(text)
            self._items.insert(idx, row)
            return idx

        def SetItem(self, idx, col, text):
            while idx >= len(self._items):
                self._items.append(["" for _ in range(max(1, len(self._columns)))])
            row = self._items[idx]
            if col >= len(row):
                row.extend([""] * (col - len(row) + 1))
            row[col] = str(text)

        def Select(self, idx):
            self._selected = int(idx)

        def GetFirstSelected(self):
            return self._selected

        def GetNextItem(self, item, flags=0, state=0):
            return -1

        def Bind(self, event, handler):
            pass

        def SetToolTip(self, tip):
            pass

        def SetMinSize(self, size):
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
            self._enabled = True
            self.label = label

        def GetValue(self):
            return self._value

        def SetValue(self, value):
            self._value = bool(value)

        def Enable(self, enable=True):
            self._enabled = enable

        def IsEnabled(self):
            return self._enabled

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

        def IsEnabled(self):
            return self._enabled

        def SetLabel(self, label):
            self.label = label

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
    FONTWEIGHT_BOLD = 92
    PD_CAN_ABORT = 0x01
    PD_APP_MODAL = 0x02
    PD_REMAINING_TIME = 0x04
    PD_AUTO_HIDE = 0x08

    # Event types
    EVT_BUTTON = "EVT_BUTTON"
    EVT_CHECKBOX = "EVT_CHECKBOX"
    EVT_LIST_ITEM_SELECTED = "EVT_LIST_ITEM_SELECTED"
    EVT_CLOSE = "EVT_CLOSE"

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
