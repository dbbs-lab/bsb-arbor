"""
Arbor simulation adapter for the BSB framework
"""

from bsb.simulation import SimulationBackendPlugin

from . import devices
from .adapter import ArborAdapter
from .simulation import ArborSimulation

__version__ = "0.0.0b1"
__plugin__ = SimulationBackendPlugin(Simulation=ArborSimulation, Adapter=ArborAdapter)
