"""
Minimal wx mock for testing without wxPython.

This module provides stub implementations of wx classes
used by ThermalSim.
"""

import sys


class _WxMock:
    """Mock wx module."""

    ID_OK = 5100
    ID_CANCEL = 5101
    DD_DEFAULT_STYLE = 0

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
        def __init__(self, parent, label="", size=None):
            self.label = label

    class TextCtrl:
        def __init__(self, parent, value="", style=0):
            self._value = value

        def GetValue(self):
            return self._value

        def SetValue(self, value):
            self._value = str(value)

        def Enable(self, enable=True):
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

    class Button:
        def __init__(self, parent, id=None, label=""):
            self.label = label

        def Bind(self, event, handler):
            pass

    class DirDialog:
        def __init__(self, parent, message="", defaultPath="", style=0):
            self.path = defaultPath

        def ShowModal(self):
            return _WxMock.ID_OK

        def GetPath(self):
            return self.path

        def Destroy(self):
            pass

    # Constants
    VERTICAL = 1
    HORIZONTAL = 0
    ALL = 0x0F
    EXPAND = 0x01
    RIGHT = 0x02
    TE_MULTILINE = 0x01
    TE_READONLY = 0x02
    TE_DONTWRAP = 0x04
    ALIGN_CENTER_VERTICAL = 0x08

    # Event types
    EVT_BUTTON = "EVT_BUTTON"
    EVT_CHECKBOX = "EVT_CHECKBOX"


def install_wx_mock():
    """Install wx mock into sys.modules."""
    sys.modules['wx'] = _WxMock()


def uninstall_wx_mock():
    """Remove wx mock from sys.modules."""
    if 'wx' in sys.modules and isinstance(sys.modules['wx'], type(_WxMock)):
        del sys.modules['wx']
