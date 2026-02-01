"""
ThermalSim - KiCad PCB thermal simulation plugin.

A 2.5D transient thermal simulation tool for multilayer PCBs.
"""

from .thermal_plugin import ThermalPlugin

# Plugin instanziieren und registrieren
ThermalPlugin().register()