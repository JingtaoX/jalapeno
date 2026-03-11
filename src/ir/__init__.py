"""
IR package — re-exports everything so existing `from ir import ...` still works.
"""

from ir.nodes import (
    TimingConstraint,
    PortTiming,
    Resource,
    Operation,
    BundleInfo,
    LoopRegion,
    SchedulingIR,
)
from ir.builder import IRBuilder, build_ir

__all__ = [
    "TimingConstraint",
    "PortTiming",
    "Resource",
    "Operation",
    "BundleInfo",
    "LoopRegion",
    "SchedulingIR",
    "IRBuilder",
    "build_ir",
]
