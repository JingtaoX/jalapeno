"""
SDC package — re-exports everything so existing `from sdc import ...` still works.
"""

from sdc.model import ConstraintType, Constraint, SDCModel
from sdc.generator import SDCGenerator, generate_sdc

__all__ = [
    "ConstraintType",
    "Constraint",
    "SDCModel",
    "SDCGenerator",
    "generate_sdc",
]
