"""
Jalapeno Compiler - Main entry point

Compilation pipeline:
1. Parse: source text -> parse tree
2. AST: parse tree -> AST
3. IR: AST -> Scheduling IR
4. SDC Gen: IR -> SDC constraints
5. Schedule: SDC constraints -> scheduled design
6. Codegen: scheduled design -> Jalapeno output with resolved timing

Flags:
  --show {ast,ir,sdc,schedule,emit}  Print output at each stage (repeatable)
  --stop-after {ast,ir,sdc,schedule,emit}  Stop pipeline after this stage
"""

import argparse
import json
from pathlib import Path
from parser import create_parser
from ast_builder import build_ast
from ir import build_ir
from sdc import generate_sdc
from solver import solve_sdc, SolveStatus
from codegen import generate_code
from ast_nodes import Component


# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "modules.json"

# Pipeline stage ordering (used to check stop-after)
STAGES = ["ast", "ir", "sdc", "schedule", "emit"]


def load_module_config(config_path: Path = None) -> dict:
    """Load module latencies from config file."""
    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            config = json.load(f)
        return {name: info["latency"] for name, info in config.get("modules", {}).items()}
    return {"Add": 1, "Sub": 1, "Mult": 3, "Div": 10}


def parse(source_path: str):
    """Step 1: Parse source file into parse tree"""
    parser = create_parser()
    tree = parser.parse_file(source_path)
    return tree


def build(parse_tree):
    """Step 2: Build AST from parse tree"""
    return build_ast(parse_tree)


def lower_to_ir(ast, module_latencies=None):
    """Step 3: Lower AST to Scheduling IR"""
    latencies = module_latencies or load_module_config()
    irs = []
    for module in ast.modules:
        if isinstance(module, Component):
            ir = build_ir(module, latencies)
            irs.append(ir)
    return irs


def compile(source_path: str, show: set = None, stop_after: str = "",
            verbose: bool = False):
    """Run the full compilation pipeline."""
    show = show or set()
    stop_idx = STAGES.index(stop_after) if stop_after else len(STAGES) - 1

    print(f"Compiling: {source_path}")

    # Step 1: Parse
    print("  [1/6] Parsing...")
    tree = parse(source_path)
    if verbose:
        print("--- Parse Tree ---")
        print(tree.pretty())

    # Step 2: Build AST
    print("  [2/6] Building AST...")
    ast = build(tree)
    if "ast" in show:
        print("\n" + "=" * 60)
        print("AST")
        print("=" * 60)
        print_ast(ast)
    if stop_idx <= STAGES.index("ast"):
        print("\n  Done.")
        return ast, [], [], []

    # Step 3: Build IR
    print("  [3/6] Building IR...")
    irs = lower_to_ir(ast)
    if "ir" in show:
        print("\n" + "=" * 60)
        print("Scheduling IR")
        print("=" * 60)
        for ir in irs:
            print_ir(ir)
    if stop_idx <= STAGES.index("ir"):
        print("\n  Done.")
        return ast, irs, [], []

    # Step 4: Generate SDC constraints
    print("  [4/6] Generating SDC constraints...")
    sdc_models = [generate_sdc(ir) for ir in irs]
    if "sdc" in show:
        print("\n" + "=" * 60)
        print("SDC Constraints")
        print("=" * 60)
        for model in sdc_models:
            model.print_summary()
    if stop_idx <= STAGES.index("sdc"):
        print("\n  Done.")
        return ast, irs, sdc_models, []

    # Step 5: Solve (schedule)
    print("  [5/6] Solving schedule...")
    schedules = []
    for model in sdc_models:
        sched = solve_sdc(model)
        schedules.append(sched)
    if "schedule" in show or "emit" in show:
        print("\n" + "=" * 60)
        print("Schedule Result")
        print("=" * 60)
        for sched in schedules:
            sched.print_schedule()
    if stop_idx <= STAGES.index("schedule"):
        print("\n  Done.")
        return ast, irs, sdc_models, schedules

    # Step 6: Codegen
    print("  [6/6] Generating scheduled code...")
    if "emit" in show:
        print("\n" + "=" * 60)
        print("Scheduled Output")
        print("=" * 60)
        for ir, sched in zip(irs, schedules):
            code = generate_code(ir, sched)
            print(code)

    print("\n  Done.")
    return ast, irs, sdc_models, schedules


def print_ast(ast, indent=0):
    """Pretty print AST for debugging"""
    prefix = "  " * indent
    if hasattr(ast, '__dataclass_fields__'):
        print(f"{prefix}{ast.__class__.__name__}(")
        for field in ast.__dataclass_fields__:
            value = getattr(ast, field)
            print(f"{prefix}  {field}=", end="")
            if isinstance(value, list):
                if len(value) == 0:
                    print("[]")
                else:
                    print("[")
                    for item in value:
                        print_ast(item, indent + 2)
                    print(f"{prefix}  ]")
            elif hasattr(value, '__dataclass_fields__'):
                print()
                print_ast(value, indent + 2)
            else:
                print(f"{value!r}")
        print(f"{prefix})")
    else:
        print(f"{prefix}{ast!r}")


def print_ir(ir):
    """Pretty print Scheduling IR"""
    print(f"\nComponent: {ir.name}")

    ii_str = str(ir.ii) if ir.ii is not None else "? (scheduler decides)"
    print(f"  Event: '{ir.start_event}, II={ii_str}")
    print(f"  Scheduler variables: {ir.sched_vars}")

    print("\n  Interface:")
    print("    Inputs:")
    for name, port in ir.inputs.items():
        timing_str = _format_port_timing(port, ir.start_event)
        width_str = f" {port.width}" if port.width else ""
        bundle_str = f"[{port.bundle_size}]" if port.bundle_size is not None else ""
        print(f"      {name}{bundle_str}: {timing_str}{width_str}")
    print("    Outputs:")
    for name, port in ir.outputs.items():
        timing_str = _format_port_timing(port, ir.start_event)
        width_str = f" {port.width}" if port.width else ""
        bundle_str = f"[{port.bundle_size}]" if port.bundle_size is not None else ""
        print(f"      {name}{bundle_str}: {timing_str}{width_str}")

    if ir.const_values:
        print("\n  Constants:")
        for name, val in ir.const_values.items():
            print(f"    {name} = #{val}")

    if ir.bundle_inits:
        print("\n  Wire Connects:")
        for dest, src in ir.bundle_inits.items():
            print(f"    {dest} = {src}")

    if ir.pools:
        print("\n  Resource Pools:")
        for name, pool in ir.pools.items():
            max_str = str(pool.max_instances) if pool.max_instances is not None else "unbounded"
            print(f"    {name}: {pool.module}[{_format_params(pool.params)}] * {max_str}, latency={pool.latency}")

    print("\n  Operations:")
    for op_name, op in ir.operations.items():
        timing_str = _format_timing(op.timing)
        binding_str = ""
        if op.binding:
            if op.binding.kind == 'anon':
                binding_str = "[?]"
            elif op.binding.kind == 'named':
                binding_str = f"[?{op.binding.value}]"
            elif op.binding.kind == 'explicit':
                val = op.binding.value
                if hasattr(val, 'value'):
                    val = val.value
                binding_str = f"[{val}]"
        print(f"    {op_name}: {op.result} := {op.resource}{binding_str}<{timing_str}>({', '.join(op.inputs)})")

    data_deps = _compute_data_dependencies(ir)
    if data_deps:
        print("\n  Data Dependencies:")
        for (producer, consumer) in sorted(data_deps.keys()):
            print(f"    {producer} -> {consumer}")

    if ir.loops:
        print("\n  Loop Regions:")
        for idx, region in enumerate(ir.loops):
            ii_str = str(region.ii) if region.ii is not None else "? (solver decides)"
            print(f"    Loop {idx}: for {region.loop_var} in 0..{region.trip_count}, II={ii_str}")
            if region.body_instances:
                print(f"      Instances: {dict(region.body_instances)}")
            if region.body_pools:
                print("      Pools:")
                for name, pool in region.body_pools.items():
                    max_str = str(pool.max_instances) if pool.max_instances is not None else "unbounded"
                    print(f"        {name}: {pool.module}[{_format_params(pool.params)}] * {max_str}, latency={pool.latency}")
            if region.sched_vars:
                print(f"      Sched vars: {region.sched_vars}")
            print("      Body ops:")
            for op_name, op in region.body_ops.items():
                timing_str = _format_timing(op.timing)
                binding_str = ""
                if op.binding:
                    if op.binding.kind == 'anon':
                        binding_str = "[?]"
                    elif op.binding.kind == 'named':
                        binding_str = f"[?{op.binding.value}]"
                    elif op.binding.kind == 'explicit':
                        val = op.binding.value
                        if hasattr(val, 'value'):
                            val = val.value
                        binding_str = f"[{val}]"
                print(f"        {op_name}: {op.result} := {op.resource}{binding_str}<{timing_str}>({', '.join(op.inputs)})")
            if region.body_connects:
                print("      Wire connects:")
                for dest, src in region.body_connects:
                    print(f"        {dest} = {src}")


def _compute_data_dependencies(ir):
    deps = {}
    for op_name, op in ir.operations.items():
        for inp in op.inputs:
            producer_result = inp.split('.')[0] if '.' in inp else inp
            if producer_result in ir.result_to_op:
                producer_op_name = ir.result_to_op[producer_result]
                producer_op = ir.operations[producer_op_name]
                latency = _get_latency(ir, producer_op)
                deps[(producer_op_name, op_name)] = latency
    return deps


def _get_latency(ir, op):
    resource = op.resource
    if resource in ir.pools:
        return ir.pools[resource].latency
    if resource in ir.module_latencies:
        return ir.module_latencies[resource]
    return 1


def _format_timing(timing):
    parts = []
    if timing.base_event:
        parts.append(f"'{timing.base_event}")
    elif timing.sched_var:
        parts.append(f"?{timing.sched_var}")
    elif timing.is_anonymous():
        parts.append("?")
    if timing.offset is not None:
        parts.append(f"+{timing.offset}")
    elif timing.range_lo is not None or timing.range_hi is not None:
        lo = timing.range_lo if timing.range_lo is not None else ""
        hi = timing.range_hi if timing.range_hi is not None else ""
        parts.append(f"+[{lo}..{hi}]")
    return "".join(parts)


def _format_port_timing(port, default_event="G"):
    if port.is_interface:
        event = port.start_event or "?"
        return f"interface['{event}]"
    start = _format_time_component(port.start_event, port.start_sched_var, port.start_offset, default_event)
    end = _format_time_component(port.end_event, port.end_sched_var, port.end_offset, default_event)
    return f"[{start}, {end}]"


def _format_time_component(event, sched_var, offset, default_event="G"):
    parts = []
    if event:
        parts.append(f"'{event}")
    elif sched_var is not None:
        parts.append("?" if sched_var == "" else f"?{sched_var}")
    else:
        parts.append("?")
    if offset is not None:
        parts.append(f"+{offset}" if offset >= 0 else str(offset))
    return "".join(parts)


def _format_params(params):
    if not params:
        return ""
    formatted = []
    for p in params:
        formatted.append(str(p.value) if hasattr(p, 'value') else str(p))
    return ", ".join(formatted)


def main():
    argparser = argparse.ArgumentParser(
        description="Jalapeno Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s foo.fil                           # run full pipeline silently
  %(prog)s foo.fil --show ast                # print AST
  %(prog)s foo.fil --show ir sdc            # print IR and SDC constraints
  %(prog)s foo.fil --show emit              # print scheduled output
  %(prog)s foo.fil --stop-after ir          # run only parse + AST + IR
  %(prog)s foo.fil --show ast --stop-after ast  # print AST then stop
        """,
    )
    argparser.add_argument("source", help="Source file to compile")
    argparser.add_argument(
        "--show",
        nargs="+",
        choices=STAGES,
        default=[],
        metavar="STAGE",
        help="Print output at these stages: ast ir sdc schedule emit",
    )
    argparser.add_argument(
        "--stop-after",
        choices=STAGES,
        metavar="STAGE",
        help="Stop pipeline after this stage: ast ir sdc schedule emit",
    )
    argparser.add_argument("-v", "--verbose", action="store_true", help="Print parse tree")
    argparser.add_argument("--config", type=Path, help="Module config JSON file")
    args = argparser.parse_args()

    compile(args.source, show=set(args.show), stop_after=args.stop_after,
            verbose=args.verbose)


if __name__ == "__main__":
    main()
