"""ThermalSim KiCad ActionPlugin registration entry point."""

import os
import site


# KiCad's embedded Python may disable packages installed with pip --user.
try:
    site.addsitedir(site.getusersitepackages())
except Exception:
    pass


_HEADLESS = os.environ.get("THERMALSIM_HEADLESS", "").strip().lower() in {
    "1", "true", "yes", "on",
}


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
                from .capabilities import get_missing_packages, get_pypardiso_optional_dependency

                missing = get_missing_packages()
                if not missing:
                    wx.MessageBox(
                        "All packages appear installed. Please restart KiCad.",
                        "ThermalSim",
                    )
                    return
                from .dependency_installer import DependencyInstallDialog

                dialog = DependencyInstallDialog(
                    None,
                    missing,
                    [get_pypardiso_optional_dependency()],
                )
                dialog.ShowModal()
                dialog.Destroy()

        _StubThermalPlugin().register()
