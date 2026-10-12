"""Multi-Variable Conformal Prediction (MCP).

Conformal predictors that calibrate one (SCP) or several (RemMCP, H-RelMCP)
prediction-set parameters with finite-sample coverage guarantees.
"""

from MCP.CP import CP
from MCP.HRelMCP import HRelMCP
from MCP.RemMCP import RemMCP
from MCP.SCP import SCP

__all__ = ["CP", "SCP", "RemMCP", "HRelMCP"]
