"""
pystocoptim is a package for solving Stochastic Optimal Control Problems (SOCPs) using the RPHA algorithm. This package
requires the package pyopticontrol to solve optimal control problems.
"""

__version__ = "1.0"

from .FiltrationFuns import (
    Atom, compute_filtration, get_path_values, retrieve_scenarios_from_filtration_bfs,
    retrieve_scenarios_from_filtration_dfs
)
from .RPHAFuns import SOCRPHA, RPHASolver
from .SDAPFuns import SOCSDAP, SDAPSolver
from .UtilFunctions import SDAPData, RPHAData, ScenarioData, ipms_socp
