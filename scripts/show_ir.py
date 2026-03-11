#!/usr/bin/env python3
"""
Show IR for a Jalapeno file.
Usage: python scripts/show_ir.py examples/three_add.fil
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser import create_parser
from ast_builder import build_ast
from ir import build_ir


def show_ir(filepath: str):
    """Parse a file and display its IR."""
    parser = create_parser()

    with open(filepath) as f:
        code = f.read()

    tree = parser.parse(code)
    ast = build_ast(tree)

    for module in ast.modules:
        ir = build_ir(module, module_latencies={"Add": 1, "Mult": 3})

        print(f"=" * 60)
        print(f"Component: {ir.name}")
        print(f"Start event: '{ir.start_event}, delay: {ir.start_delay}")
        print(f"Scheduler variables: {ir.sched_vars}")
        print()

        print("Resources:")
        for name, res in ir.resources.items():
            pool_info = f" (pool, max={res.max_instances})" if res.is_pool else ""
            print(f"  {name}: {res.module}, latency={res.latency}{pool_info}")
        print()

        print("Operations:")
        for name, op in ir.operations.items():
            timing_str = ""
            if op.timing.sched_var:
                timing_str = f"?{op.timing.sched_var}"
            elif op.timing.base_event:
                timing_str = f"'{op.timing.base_event}"

            if op.timing.offset:
                timing_str += f"+{op.timing.offset}"
            elif op.timing.range_lo is not None or op.timing.range_hi is not None:
                lo = op.timing.range_lo if op.timing.range_lo is not None else ""
                hi = op.timing.range_hi if op.timing.range_hi is not None else ""
                timing_str += f"+[{lo}..{hi}]"

            deps_str = f" <- {op.depends_on}" if op.depends_on else ""
            print(f"  {name}: {op.resource}<{timing_str}>({', '.join(op.inputs)}){deps_str}")
        print()

        print("Data Edges (for SDC):")
        for consumer, edges in ir.data_edges.items():
            for producer, latency in edges:
                print(f"  {consumer} depends on {producer} (latency={latency})")
                print(f"    => t_{consumer} >= t_{producer} + {latency}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/show_ir.py <file.fil>")
        sys.exit(1)

    show_ir(sys.argv[1])
