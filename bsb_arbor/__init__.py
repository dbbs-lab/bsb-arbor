"""
Arbor simulation adapter for the BSB framework
"""

from bsb.simulation import SimulationBackendPlugin
from .simulation import ArborSimulation
from .adapter import ArborAdapter
from . import devices

__version__ = "0.0.0a0"
__plugin__ = SimulationBackendPlugin(Simulation=ArborSimulation, Adapter=ArborAdapter)
