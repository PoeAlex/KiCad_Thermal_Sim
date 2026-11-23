"""Module execution entrypoint for ``python3 -m KiCad_Thermal_Sim``."""

import os
import sys

os.environ.setdefault("THERMALSIM_HEADLESS", "1")

from .thermalsim_cli import main


if __name__ == "__main__":
    sys.exit(main())
