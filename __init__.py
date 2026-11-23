"""ThermalSim KiCad ActionPlugin registration entry point."""

import os
import site
import sys


# KiCad's embedded Python may disable packages installed with pip --user.
try:
    site.addsitedir(site.getusersitepackages())
except Exception:
    pass


_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_argv0 = sys.argv[0] if getattr(sys, "argv", None) else ""
_argv0_name = os.path.basename(_argv0) if _argv0 else ""
_CLI_INVOCATION = (
    _argv0 == "-m"
    or _argv0_name == "thermalsim_cli.py"
    or (
        _argv0_name == "__main__.py"
        and (_argv0 == "__main__.py" or os.path.dirname(os.path.abspath(_argv0)) == _PKG_DIR)
    )
)
if _CLI_INVOCATION:
    os.environ.setdefault("THERMALSIM_HEADLESS", "1")

_HEADLESS = _CLI_INVOCATION or (
    os.environ.get("THERMALSIM_HEADLESS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
)


def _register_plugin():
    """Register the full plugin."""
    from .thermal_plugin import ThermalPlugin
    ThermalPlugin().register()


if not _HEADLESS:
    try:
        _register_plugin()
    except ImportError:
        # Core dependencies missing: register an installer fallback.
        import pcbnew

        class _StubThermalPlugin(pcbnew.ActionPlugin):
            """Fallback plugin that opens the dependency installer."""

            def defaults(self):
                self.name = "2.5D Thermal Sim"
                self.category = "Simulation"
                self.description = "Crash-safe Multilayer Sim (dependencies missing)"
                self.show_toolbar_button = True
                self.icon_file_name = os.path.join(os.path.dirname(__file__), "ThermalSim_icon.png")

            def Run(self):
                import wx
                from .capabilities import get_missing_packages

                missing = get_missing_packages()
                if not missing:
                    wx.MessageBox(
                        "All packages appear installed. Please restart KiCad.",
                        "ThermalSim",
                    )
                    return
                from .dependency_installer import DependencyInstallDialog

                dialog = DependencyInstallDialog(None, missing)
                dialog.ShowModal()
                dialog.Destroy()

        _StubThermalPlugin().register()
