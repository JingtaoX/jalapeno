"""
Jalapeno Parser

Parses Filament/Jalapeno source code into a parse tree using Lark.
The parse tree is then transformed into an AST by ast_builder.py.
"""

from pathlib import Path
from lark import Lark

# Re-export AST nodes for backward compatibility
from ast_nodes import (
    # Expressions
    Number, Var, BinOp,
    # Time system
    Event, SchedVar, RangeOffset, Time, Interval, EventBind,
    # Ports
    PortDef, PortRef,
    # Module structure
    Param, Constraint, Signature,
    # Resource pools
    Pool, PoolSize, BindingIndex,
    # Commands
    Instance, Invocation, Connect, TimeLoop, SpaceLoop, IfStmt, ParamLet,
    # Top-level
    Component, ExternBlock, Import, File,
)


class JalapenoParser:
    """
    Parser for Jalapeno/Filament source code.

    Uses Lark with Earley parser to handle the grammar's context-sensitive
    constructs (e.g., < > used for both generics and comparisons).
    """

    def __init__(self):
        grammar_path = Path(__file__).parent.parent / "grammar" / "jalapeno.lark"
        with open(grammar_path) as f:
            grammar = f.read()

        self.parser = Lark(
            grammar,
            start='start',
            parser='earley',
            ambiguity='resolve',
        )

    def parse(self, text: str):
        """Parse source code and return parse tree."""
        return self.parser.parse(text)

    def parse_file(self, path: str):
        """Parse a source file and return parse tree."""
        with open(path) as f:
            return self.parse(f.read())


def create_parser() -> JalapenoParser:
    """Factory function to create a parser instance."""
    return JalapenoParser()
