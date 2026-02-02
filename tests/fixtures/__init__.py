"""
Test fixtures for ThermalSim.

This package provides test data generators and configurations
for unit tests.
"""

from .sample_boards import (
    SIMPLE_2_LAYER_STACKUP,
    SIMPLE_4_LAYER_STACKUP,
    generate_kicad_pcb_content,
)
from .stackup_configs import (
    STACKUP_2_LAYER,
    STACKUP_4_LAYER,
    STACKUP_6_LAYER,
)
from .temperature_arrays import (
    create_uniform_temperature,
    create_gradient_temperature,
    create_hotspot_temperature,
)
