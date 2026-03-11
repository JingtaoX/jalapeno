"""
Codegen: Output scheduled program in Jalapeno/Filament syntax

Takes the IR and Schedule, outputs the program with all scheduler variables
(?, ?name, ?name+offset, ?name+[lo..hi]) replaced with concrete values.

Example transformation:
  Input:  b1 := A[?]<?>(a1, a2);
  Output: b1 := A[0]<'G+0>(a1, a2);

  Input:  o := A[?]<?t_b2+[..2]>(b1.out, b2.out);
  Output: o := A[0]<'G+1>(b1.out, b2.out);

Loop regions are emitted as:
  timeloop<?tloop='G> [II=<resolved>] for i in 0..<N> {
    <body ops with resolved timing relative to loop start>
    <body connects>
  }
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from ir import SchedulingIR, Operation, Resource, PortTiming, LoopRegion
from solver import Schedule, SolveStatus


@dataclass
class CodegenConfig:
    """Configuration for code generation."""
    event_name: str = "G"
    include_comments: bool = True
    indent: str = "  "


class Codegen:
    """
    Generate scheduled Jalapeno/Filament code from IR and Schedule.
    """

    def __init__(self, ir: SchedulingIR, schedule: Schedule, config: CodegenConfig = None):
        self.ir = ir
        self.schedule = schedule
        self.config = config or CodegenConfig()

        self.op_timing: Dict[str, int] = {}
        self.op_binding: Dict[str, int] = {}
        self.pool_sizes: Dict[str, int] = {}  # pool_name -> solved size
        self._extract_assignments()

    def _extract_assignments(self):
        """Extract timing, binding, and pool size assignments from schedule."""
        for var, val in self.schedule.assignments.items():
            if var.endswith('.t'):
                op_name = var[:-2]
                self.op_timing[op_name] = val
            elif var.endswith('.i'):
                op_name = var[:-2]
                self.op_binding[op_name] = val

        # Compute solved pool sizes: max(op.i)+1 over all ops on each pool.
        all_pools = {**self.ir.pools}
        for region in self.ir.loops:
            all_pools.update(region.body_pools)
        pool_max_unit: Dict[str, int] = {}
        all_ops = {**self.ir.operations}
        for region in self.ir.loops:
            all_ops.update(region.body_ops)
        for op_name, op in all_ops.items():
            if op.resource in all_pools:
                unit = self.op_binding.get(op_name, 0)
                pool_max_unit[op.resource] = max(pool_max_unit.get(op.resource, 0), unit)
        for pool_name in all_pools:
            if pool_name in pool_max_unit:
                self.pool_sizes[pool_name] = pool_max_unit[pool_name] + 1

    def generate(self) -> str:
        """Generate the complete scheduled program."""
        if self.schedule.status not in (SolveStatus.OPTIMAL, SolveStatus.SATISFIABLE):
            return f"// Schedule failed: {self.schedule.status.value}\n// {self.schedule.message}"

        lines = []

        lines.append(f"// Scheduled output for component: {self.ir.name}")
        lines.append(f"// Makespan: {self.schedule.makespan}")
        lines.append("")

        lines.append('import "primitives/core.fil";')
        lines.append("")

        lines.append(self._gen_signature())
        lines.append("{")

        # Top-level pool declarations
        for pool_name, pool in self.ir.pools.items():
            lines.append(self._gen_pool_decl(pool_name, pool))
        if self.ir.pools:
            lines.append("")

        # Top-level bundle declarations
        for bundle_name, bundle in self.ir.bundles.items():
            lines.append(self._gen_bundle_decl(bundle_name, bundle))
        if self.ir.bundles:
            lines.append("")

        # Bundle element inits (e.g. s{0} = #0) — before loops that read them
        indent = self.config.indent
        for dest, src in self.ir.bundle_inits.items():
            lines.append(f"{indent}{dest} = {src};")
        if self.ir.bundle_inits:
            lines.append("")

        # Loop regions first (they start at the component's start event)
        for idx, region in enumerate(self.ir.loops):
            lines.extend(self._gen_loop_region(region, idx))
            lines.append("")

        # Flat operations after loops (post-loop ops fire later in time)
        for op_name, op in self.ir.operations.items():
            lines.append(self._gen_invocation(op_name, op, indent=self.config.indent))

        # Top-level output connections (op-result: r.out -> output)
        for out_name, (producer_op, port) in self.ir.output_connections.items():
            lines.append(self._gen_output_connection(out_name, producer_op, port))

        # Top-level bundle connections (whole-bundle: sum = s)
        indent = self.config.indent
        for out_name, src_bundle in self.ir.bundle_connections.items():
            lines.append(f"{indent}{out_name} = {src_bundle};")

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Signature
    # ------------------------------------------------------------------

    def _gen_signature(self) -> str:
        event = self.ir.start_event
        ii = self.ir.ii if self.ir.ii is not None else self.schedule.makespan

        inputs = [self._gen_port_def(n, p) for n, p in self.ir.inputs.items()]
        outputs = [self._gen_port_def(n, p, is_output=True) for n, p in self.ir.outputs.items()]

        indent = self.config.indent
        sig = f"comp {self.ir.name}<'{event}: {ii}>(\n"
        sig += f",\n".join(f"{indent}{p}" for p in inputs)
        sig += f"\n) -> ("
        sig += ", ".join(outputs)
        sig += ")"
        return sig

    def _gen_port_def(self, name: str, port: PortTiming, is_output: bool = False) -> str:
        event = self.ir.start_event
        width = port.width or 32
        makespan = self.schedule.makespan or 0

        bundle_suffix = f"[{port.bundle_size}]" if port.bundle_size else ""

        def _fmt_time(ev, sv, off, fill_with_makespan):
            if ev:
                s = f"'{ev}"
                if off:
                    s += f"+{off}"
                return s
            if sv is not None:
                # Anonymous ? -> fill with makespan if requested
                if fill_with_makespan:
                    return f"'{event}+{makespan}"
                return "?"
            return f"'{event}"

        # Input: start is fixed ('G), end ? -> fill with makespan
        # Output: start ? -> makespan, end ? -> makespan+1 (one-cycle output window)
        start = _fmt_time(port.start_event, port.start_sched_var, port.start_offset,
                          fill_with_makespan=is_output)
        if is_output and port.end_sched_var is not None and not port.end_event:
            end = f"'{event}+{makespan + 1}"
        else:
            end = _fmt_time(port.end_event, port.end_sched_var, port.end_offset,
                            fill_with_makespan=True)

        return f"{name}{bundle_suffix}: [{start}, {end}] {width}"

    # ------------------------------------------------------------------
    # Pool / bundle declarations
    # ------------------------------------------------------------------

    def _gen_pool_decl(self, name: str, pool: Resource, extra_indent: str = "") -> str:
        indent = self.config.indent + extra_indent
        params = self._format_params(pool.params)
        if name in self.pool_sizes:
            size_str = str(self.pool_sizes[name])
        elif pool.max_instances is not None:
            size_str = str(pool.max_instances)
        else:
            size_str = "?"
        return f"{indent}pool {name} : {pool.module}[{params}] * {size_str};"

    def _gen_bundle_decl(self, name: str, bundle, extra_indent: str = "") -> str:
        indent = self.config.indent + extra_indent
        event = self.ir.start_event
        width = bundle.width or 32
        start = f"'{bundle.start_event}" if bundle.start_event else f"'{event}"
        if bundle.start_offset:
            start += f"+{bundle.start_offset}"
        end = f"'{bundle.end_event}" if bundle.end_event else "?"
        if bundle.end_offset:
            end += f"+{bundle.end_offset}"
        return f"{indent}bundle {name}[{bundle.bundle_size}]: [{start}, {end}] {width};"

    # ------------------------------------------------------------------
    # Invocations
    # ------------------------------------------------------------------

    def _gen_invocation(self, op_name: str, op: Operation,
                        indent: str = None, loop_start_time: int = 0) -> str:
        if indent is None:
            indent = self.config.indent
        event = self.ir.start_event

        # Timing: absolute time from schedule, expressed as offset from event
        abs_time = self.op_timing.get(op_name, 0)
        rel_time = abs_time - loop_start_time

        # Binding
        binding_str = ""
        all_pools = {**self.ir.pools}
        for region in self.ir.loops:
            all_pools.update(region.body_pools)
        if op.resource in all_pools:
            binding = self.op_binding.get(op_name, 0)
            binding_str = f"[{binding}]"

        if rel_time == 0:
            timing_str = f"'{event}"
        else:
            timing_str = f"'{event}+{rel_time}"

        inputs_str = ", ".join(op.inputs)
        line = f"{indent}{op.result} := {op.resource}{binding_str}<{timing_str}>({inputs_str});"

        if self.config.include_comments:
            original = self._format_original_constraint(op)
            if original:
                line += f"  // was: {original}"

        return line

    def _format_original_constraint(self, op: Operation) -> str:
        parts = []
        if op.binding:
            if op.binding.kind == 'anon':
                parts.append("[?]")
            elif op.binding.kind == 'named':
                parts.append(f"[?{op.binding.value}]")
            elif op.binding.kind == 'explicit':
                val = op.binding.value
                if hasattr(val, 'value'):
                    val = val.value
                parts.append(f"[{val}]")

        timing = op.timing
        if timing.alias:
            t = f"?{timing.alias}"
            if timing.sched_var:
                t = f"?{timing.alias}=?{timing.sched_var}"
            elif timing.base_event:
                t = f"?{timing.alias}='{timing.base_event}"
                if timing.offset:
                    t += f"+{timing.offset}"
            parts.append(f"<{t}>")
        elif timing.loop_end and timing.sched_var:
            t = f"?{timing.sched_var}.e"
            if timing.offset:
                t += f"+{timing.offset}"
            parts.append(f"<{t}>")
        elif timing.sched_var:
            t = f"?{timing.sched_var}"
            if timing.offset:
                t += f"+{timing.offset}"
            elif timing.range_lo is not None or timing.range_hi is not None:
                lo = timing.range_lo if timing.range_lo is not None else ""
                hi = timing.range_hi if timing.range_hi is not None else ""
                t += f"+[{lo}..{hi}]"
            parts.append(f"<{t}>")
        elif timing.base_event:
            t = f"'{timing.base_event}"
            if timing.offset:
                t += f"+{timing.offset}"
            parts.append(f"<{t}>")
        elif timing.is_anonymous():
            parts.append("<?>")

        return "".join(parts)

    # ------------------------------------------------------------------
    # Loop region
    # ------------------------------------------------------------------

    def _gen_loop_region(self, region: LoopRegion, idx: int) -> List[str]:
        lines = []
        indent = self.config.indent
        inner_indent = indent + indent
        event = self.ir.start_event

        # Resolve II
        if region.ii is not None:
            resolved_ii = region.ii
        else:
            resolved_ii = self.schedule.assignments.get(f"Loop.{idx}.II", "?")

        # Loop start alias
        start = region.start
        if start.alias:
            start_str = f"?{start.alias}='{start.base_event or event}"
            if start.offset:
                start_str += f"+{start.offset}"
        elif start.base_event:
            start_str = f"'{start.base_event}"
            if start.offset:
                start_str += f"+{start.offset}"
        else:
            start_str = f"'{event}"

        lines.append(
            f"{indent}timeloop<{start_str}> [II={resolved_ii}] "
            f"for {region.loop_var} in 0..{region.trip_count} {{"
        )

        # Pool declarations inside loop body
        for pool_name, pool in region.body_pools.items():
            lines.append(self._gen_pool_decl(pool_name, pool, extra_indent=indent))

        # Bundle declarations inside loop body
        for bundle_name, bundle in region.body_bundles.items():
            lines.append(self._gen_bundle_decl(bundle_name, bundle, extra_indent=indent))

        if region.body_pools or region.body_bundles:
            lines.append("")

        # Body invocations — timing relative to loop start (0)
        # Loop start resolves to absolute time 0 (all body ops scheduled in [0, II))
        for op_name, op in region.body_ops.items():
            lines.append(
                self._gen_invocation(op_name, op, indent=inner_indent, loop_start_time=0)
            )

        # Body connect statements (intra-body wire assignments)
        if region.body_connects:
            lines.append("")
            for dest_str, src_str in region.body_connects:
                lines.append(f"{inner_indent}{dest_str} = {src_str};")

        lines.append(f"{indent}}}")
        return lines

    # ------------------------------------------------------------------
    # Output connections
    # ------------------------------------------------------------------

    def _gen_output_connection(self, out_name: str, producer_op: str, port: str) -> str:
        indent = self.config.indent
        # Check flat ops first, then loop body ops
        producer = self.ir.operations.get(producer_op)
        if producer:
            return f"{indent}{out_name} = {producer.result}.{port};"
        for region in self.ir.loops:
            producer = region.body_ops.get(producer_op)
            if producer:
                return f"{indent}{out_name} = {producer.result}.{port};"
        return f"{indent}{out_name} = ???.{port};  // ERROR: unknown producer"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_params(self, params: List) -> str:
        if not params:
            return ""
        formatted = []
        for p in params:
            if hasattr(p, 'value'):
                formatted.append(str(p.value))
            else:
                formatted.append(str(p))
        return ", ".join(formatted)


def generate_code(ir: SchedulingIR, schedule: Schedule, config: CodegenConfig = None) -> str:
    """Generate scheduled Jalapeno code from IR and Schedule."""
    return Codegen(ir, schedule, config).generate()
