"""
Auto-dependency installer dialog for ThermalSim.

This module provides a wxPython dialog that detects missing Python packages
and offers to install them via pip. It is designed to work without numpy,
scipy, or matplotlib being installed.

Only imports wx, sys, subprocess, and threading — all available in KiCad's
Python environment regardless of whether scientific packages are installed.
"""

import sys
import subprocess
import threading

import wx


def _find_python():
    """Find the Python interpreter for pip commands.

    In embedded environments (e.g. KiCad), sys.executable points to the
    host application (kicad.exe / pcbnew.exe) rather than python.exe.
    This function locates the actual Python interpreter.
    """
    import os

    exe = sys.executable
    # If sys.executable is already Python, use it directly
    if "python" in os.path.basename(exe).lower():
        return exe

    # Look for python.exe in the same directory as the host application
    # (KiCad ships python.exe alongside kicad.exe in the bin/ folder)
    exe_dir = os.path.dirname(exe)
    for name in ("python.exe", "python3.exe", "python"):
        candidate = os.path.join(exe_dir, name)
        if os.path.isfile(candidate):
            return candidate

    # Try sys.prefix (Python installation root)
    for name in ("python.exe", "python3.exe", "python"):
        candidate = os.path.join(sys.prefix, name)
        if os.path.isfile(candidate):
            return candidate

    # Last resort: fall back to sys.executable
    return exe


class DependencyInstallDialog(wx.Dialog):
    """
    Dialog that shows missing packages and installs them via pip.

    Parameters
    ----------
    parent : wx.Window or None
        Parent window.
    missing_packages : list of tuple
        List of (import_name, pip_name) tuples for missing packages.
    """

    def __init__(self, parent, missing_packages):
        super().__init__(parent, title="ThermalSim - Missing Dependencies",
                         size=(520, 400))
        self._missing = list(missing_packages)
        self._process = None
        self._installing = False

        self._build_ui()
        self.Center()

    def _build_ui(self):
        """Build the dialog UI."""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Header
        header = wx.StaticText(panel, label="ThermalSim requires additional packages")
        font = header.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        header.SetFont(font)
        vbox.Add(header, flag=wx.ALL, border=10)

        # Missing packages list
        pkg_names = ", ".join(pip for _, pip in self._missing)
        info = wx.StaticText(
            panel,
            label=f"Missing: {pkg_names}\n\n"
                  f"Click 'Install Now' to install into KiCad's Python environment."
        )
        vbox.Add(info, flag=wx.LEFT | wx.RIGHT, border=10)

        # Output log
        self._log = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
        )
        vbox.Add(self._log, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)

        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._btn_install = wx.Button(panel, label="Install Now")
        self._btn_install.Bind(wx.EVT_BUTTON, self._on_install)
        btn_sizer.Add(self._btn_install, flag=wx.RIGHT, border=5)

        self._btn_close = wx.Button(panel, id=wx.ID_CANCEL, label="Close")
        btn_sizer.Add(self._btn_close)

        vbox.Add(btn_sizer, flag=wx.ALIGN_RIGHT | wx.ALL, border=10)

        # Manual instructions (hidden initially)
        self._manual_text = wx.StaticText(panel, label="")
        vbox.Add(self._manual_text, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10)

        panel.SetSizer(vbox)

    def _on_install(self, event):
        """Start pip install in a background thread."""
        if self._installing:
            return
        self._installing = True
        self._btn_install.Enable(False)
        self._log.SetValue("")
        self._append_log("Starting installation...\n")

        pip_names = [pip for _, pip in self._missing]
        cmd = [_find_python(), "-m", "pip", "install"] + pip_names

        self._append_log(f"> {' '.join(cmd)}\n\n")

        t = threading.Thread(target=self._run_pip, args=(cmd,), daemon=True)
        t.start()

    def _run_pip(self, cmd):
        """Run pip subprocess and stream output."""
        try:
            kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
            }
            # On Windows, hide the console window
            if sys.platform == "win32":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 0  # SW_HIDE
                kwargs["startupinfo"] = si

            self._process = subprocess.Popen(cmd, **kwargs)

            for line in self._process.stdout:
                wx.CallAfter(self._append_log, line)

            self._process.wait()
            rc = self._process.returncode
            self._process = None

            if rc == 0:
                wx.CallAfter(self._on_success)
            else:
                wx.CallAfter(self._on_failure, rc)

        except Exception as exc:
            wx.CallAfter(self._on_failure, -1, str(exc))

    def _append_log(self, text):
        """Append text to the log control (must be called on main thread)."""
        self._log.AppendText(text)

    def _on_success(self):
        """Handle successful installation."""
        self._installing = False
        self._append_log("\nInstallation successful!\n")
        self._append_log("Please restart KiCad for changes to take effect.\n")
        self._btn_install.SetLabel("Done")
        self._btn_install.Enable(False)

    def _on_failure(self, returncode, error_msg=None):
        """Handle failed installation."""
        self._installing = False
        self._append_log(f"\nInstallation failed (exit code {returncode}).\n")
        if error_msg:
            self._append_log(f"Error: {error_msg}\n")

        pip_names = " ".join(pip for _, pip in self._missing)
        manual = (
            f"Manual install: Open 'KiCad 9.0 Command Prompt' and run:\n"
            f"  pip install {pip_names}"
        )
        self._manual_text.SetLabel(manual)
        self._append_log(f"\n{manual}\n")

        self._btn_install.SetLabel("Retry")
        self._btn_install.Enable(True)
