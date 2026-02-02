"""
KiCad pcbnew mock objects for testing.

This package provides mock implementations of KiCad's pcbnew API
to allow testing without a running KiCad instance.
"""

from .pcbnew_mock import (
    MockBoard,
    MockPad,
    MockZone,
    MockFootprint,
    MockTrack,
    MockVia,
    VECTOR2I,
    EDA_RECT,
    PAD_ATTRIB_PTH,
    PAD_ATTRIB_SMD,
    Eco1_User,
    FromMM,
    ToMM,
    install_mock,
    uninstall_mock,
)

from .wx_mock import install_wx_mock, uninstall_wx_mock
