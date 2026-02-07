"""
ThermalSim - KiCad PCB thermal simulation plugin.

A 2.5D transient thermal simulation tool for multilayer PCBs.

If core dependencies (numpy, scipy, matplotlib) are missing, a stub plugin
is registered that shows an auto-install dialog instead.
"""

try:
    from .thermal_plugin import ThermalPlugin
    ThermalPlugin().register()
except ImportError:
    # Core dependencies missing — register a stub that offers to install them
    import pcbnew

    class _StubThermalPlugin(pcbnew.ActionPlugin):
        """Fallback plugin that shows the dependency installer dialog."""

        def defaults(self):
            self.name = "2.5D Thermal Sim"
            self.category = "Simulation"
            self.description = "Crash-safe Multilayer Sim (dependencies missing)"
            self.show_toolbar_button = True
            self.icon_file_name = ""

        def Run(self):
            import wx
            from .capabilities import get_missing_packages
            missing = get_missing_packages()
            if not missing:
                wx.MessageBox(
                    "All packages appear installed. Please restart KiCad.",
                    "ThermalSim"
                )
                return
            from .dependency_installer import DependencyInstallDialog
            dlg = DependencyInstallDialog(None, missing)
            dlg.ShowModal()
            dlg.Destroy()

    _StubThermalPlugin().register()
