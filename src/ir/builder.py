"""
IR Builder

Lowers a Component AST node into a SchedulingIR.

Dispatch:
  Instance    -> module_latencies entry
  Pool        -> pools entry
  Invocation  -> flat operation in ir.operations
  Connect     -> output_connections entry
  BundleDecl  -> BundleInfo in ir.bundles
  SpaceLoop   -> unrolled into flat operations (suffixed _0, _1, ...)
  TimeLoop    -> LoopRegion appended to ir.loops
  IfStmt      -> TODO
"""

from typing import Dict, List, Optional

from ast_nodes import (
    Component, Instance, Invocation, Pool, Connect,
    TimeLoop, SpaceLoop, IfStmt, BundleDecl,
    PortDef, Time, SchedVar, RangeOffset, Number, Var, BinOp,
    PortRef, PortBundleAccess,
)
from ir.nodes import (
    SchedulingIR, LoopRegion, Operation, Resource,
    PortTiming, TimingConstraint, BundleInfo,
)


class IRBuilder:
    """Builds SchedulingIR from a Component AST node."""

    def __init__(self, module_latencies: Optional[Dict[str, int]] = None):
        self.module_latencies = module_latencies or {"Add": 1}

    def build(self, component: Component) -> SchedulingIR:
        ir = SchedulingIR(name=component.signature.name)
        self._extract_event_binds(component, ir)
        self._extract_ports(component, ir)

        for cmd in component.commands:
            if isinstance(cmd, Instance):
                self._process_instance(cmd, ir)
            elif isinstance(cmd, Pool):
                self._process_pool(cmd, ir)
            elif isinstance(cmd, Invocation):
                self._process_invocation(cmd, ir)
            elif isinstance(cmd, Connect):
                self._process_connect(cmd, ir)
            elif isinstance(cmd, BundleDecl):
                self._process_bundle_decl(cmd, ir)
            elif isinstance(cmd, SpaceLoop):
                self._process_spaceloop(cmd, ir)
            elif isinstance(cmd, TimeLoop):
                self._process_timeloop(cmd, ir)
            elif isinstance(cmd, IfStmt):
                # TODO: Handle in future
                pass

        # Post-processing: wire loop_end_vars onto LoopRegions.
        # For each flat op that references ?alias.e, find the LoopRegion whose
        # start alias matches, and record the end-var name there.
        self._wire_loop_end_vars(ir)

        return ir

    def _wire_loop_end_vars(self, ir: SchedulingIR):
        """Register ?alias_end aux variables on the matching LoopRegion."""
        # Build alias -> region map from loop start aliases
        alias_to_region = {}
        for region in ir.loops:
            if region.start.alias:
                alias_to_region[region.start.alias] = region

        # Scan flat ops for loop_end references
        for op in ir.operations.values():
            if op.timing.loop_end and op.timing.sched_var:
                alias = op.timing.sched_var
                region = alias_to_region.get(alias)
                if region is not None:
                    # Find which loop index this region corresponds to
                    loop_idx = ir.loops.index(region)
                    end_var_name = f"Loop.{loop_idx}.end"
                    region.loop_end_vars[alias] = end_var_name

    # -------------------------------------------------------------------------
    # Event binds and ports
    # -------------------------------------------------------------------------

    def _extract_event_binds(self, component: Component, ir: SchedulingIR):
        for eb in component.signature.event_binds:
            ir.start_event = eb.event.name
            if isinstance(eb.delay, Number):
                ir.ii = eb.delay.value
            elif isinstance(eb.delay, SchedVar):
                ir.ii = None
                if eb.delay.name:
                    ir.sched_vars.add(eb.delay.name)

    def _extract_ports(self, component: Component, ir: SchedulingIR):
        for port_def in component.signature.inputs:
            ir.inputs[port_def.name] = self._port_def_to_timing(port_def, ir)
        for port_def in component.signature.outputs:
            ir.outputs[port_def.name] = self._port_def_to_timing(port_def, ir)

    def _port_def_to_timing(self, port_def: PortDef, ir: SchedulingIR) -> PortTiming:
        pt = PortTiming(name=port_def.name)
        pt.width = self._eval_expr(port_def.width)
        pt.bundle_size = port_def.bundle_size  # None for scalar, N for bundle

        if port_def.is_interface:
            pt.is_interface = True
            if port_def.interface_event:
                pt.start_event = port_def.interface_event.name
            return pt

        if port_def.interval:
            interval = port_def.interval

            if interval.start:
                s = interval.start
                if s.event:
                    pt.start_event = s.event.name
                if s.sched_var:
                    pt.start_sched_var = s.sched_var.name or ""
                    if s.sched_var.name:
                        ir.sched_vars.add(s.sched_var.name)
                if s.offset:
                    pt.start_offset = self._eval_expr(s.offset)

            if interval.end:
                e = interval.end
                if e.event:
                    pt.end_event = e.event.name
                if e.sched_var:
                    pt.end_sched_var = e.sched_var.name or ""
                    if e.sched_var.name:
                        ir.sched_vars.add(e.sched_var.name)
                if e.offset:
                    pt.end_offset = self._eval_expr(e.offset)

        return pt

    # -------------------------------------------------------------------------
    # Flat command processors
    # -------------------------------------------------------------------------

    def _process_instance(self, inst: Instance, ir: SchedulingIR):
        if inst.module == "Const" and len(inst.params) >= 2:
            val = self._eval_expr(inst.params[1])
            if val is not None:
                ir.const_values[inst.name] = val
                return
        ir.module_latencies[inst.name] = self.module_latencies.get(inst.module, 1)

    def _process_pool(self, pool: Pool, ir: SchedulingIR):
        latency = self.module_latencies.get(pool.module, 1)
        max_inst = None
        if pool.size and not pool.size.is_unbounded:
            max_inst = self._eval_expr(pool.size.max_count)
        ir.pools[pool.name] = Resource(
            name=pool.name,
            module=pool.module,
            params=pool.params,
            latency=latency,
            is_pool=True,
            max_instances=max_inst,
        )

    def _process_invocation(self, inv: Invocation, ir: SchedulingIR):
        resource = inv.instance
        if resource not in ir._op_counters:
            ir._op_counters[resource] = 0
        op_index = ir._op_counters[resource]
        ir._op_counters[resource] += 1
        op_name = f"{resource}.{op_index}"

        timing = self._extract_timing(inv.time_args[0] if inv.time_args else None, ir)
        inputs = self._extract_inputs(inv.args, ir.const_values)

        ir.operations[op_name] = Operation(
            op_name=op_name,
            resource=resource,
            result=inv.name,
            timing=timing,
            inputs=inputs,
            binding=inv.binding,
        )
        ir.result_to_op[inv.name] = op_name

    def _process_connect(self, conn: Connect, ir: SchedulingIR):
        # Bundle-element init: s{0} = zero  or  s{0} = #0
        if isinstance(conn.dest, PortBundleAccess) and conn.dest.instance is None:
            index_str = self._format_expr(conn.dest.index)
            dest_key = f"{conn.dest.bundle}{{{index_str}}}"
            src_str = self._format_port_expr(conn.src, ir.const_values)
            ir.bundle_inits[dest_key] = src_str
            return

        dest_name = None
        if isinstance(conn.dest, PortRef) and conn.dest.instance is None:
            dest_name = conn.dest.port
        elif isinstance(conn.dest, Var):
            dest_name = conn.dest.name

        if dest_name and dest_name in ir.outputs:
            if isinstance(conn.src, PortRef):
                result_wire = conn.src.instance
                producer_port = conn.src.port
                if result_wire and result_wire in ir.result_to_op:
                    # op-result connection: r.out -> output port
                    producer_op_name = ir.result_to_op[result_wire]
                    ir.output_connections[dest_name] = (producer_op_name, producer_port)
                elif conn.src.instance is None and conn.src.port in ir.bundles:
                    # whole-bundle connection: sum = s
                    ir.bundle_connections[dest_name] = conn.src.port
            elif isinstance(conn.src, PortBundleAccess) and conn.src.instance is None:
                # bundle-element connection: sum = s{8}
                index_str = self._format_expr(conn.src.index)
                ir.bundle_connections[dest_name] = f"{conn.src.bundle}{{{index_str}}}"

    def _process_bundle_decl(self, decl: BundleDecl, ir: SchedulingIR):
        """Record a body-level bundle declaration."""
        ir.bundles[decl.name] = self._make_bundle_info(decl)

    def _make_bundle_info(self, decl: BundleDecl) -> BundleInfo:
        """Build a BundleInfo from a BundleDecl AST node."""
        info = BundleInfo(
            name=decl.name,
            bundle_size=decl.bundle_size,
            width=self._eval_expr(decl.width),
        )
        if decl.interval:
            s = decl.interval.start
            if s.event:
                info.start_event = s.event.name
            if s.sched_var:
                info.start_sched_var = s.sched_var.name or ""
            if s.offset:
                info.start_offset = self._eval_expr(s.offset)
            e = decl.interval.end
            if e.event:
                info.end_event = e.event.name
            if e.sched_var:
                info.end_sched_var = e.sched_var.name or ""
            if e.offset:
                info.end_offset = self._eval_expr(e.offset)
        return info

    # -------------------------------------------------------------------------
    # SpaceLoop: unroll into flat IR
    # -------------------------------------------------------------------------

    def _process_spaceloop(self, loop: SpaceLoop, ir: SchedulingIR):
        """Unroll a SpaceLoop: each iteration i produces suffixed copies (_0, _1, ...)."""
        n = self._eval_expr(loop.end)
        if n is None:
            raise ValueError(f"SpaceLoop trip count must be statically known, got: {loop.end!r}")

        for i in range(n):
            suffix = f"_{i}"
            name_map: Dict[str, str] = {}
            loop_var = loop.var  # e.g. "i"

            for cmd in loop.body:
                if isinstance(cmd, Instance):
                    suffixed = Instance(
                        name=cmd.name + suffix,
                        module=cmd.module,
                        params=cmd.params,
                        time_args=cmd.time_args,
                        args=cmd.args,
                    )
                    name_map[cmd.name] = cmd.name + suffix
                    self._process_instance(suffixed, ir)

                elif isinstance(cmd, Invocation):
                    # Pools (shared resources) keep their name; only per-iter instances get suffix.
                    if cmd.instance in name_map:
                        new_instance = name_map[cmd.instance]
                    elif cmd.instance in ir.pools:
                        new_instance = cmd.instance  # shared pool — no suffix
                    else:
                        new_instance = cmd.instance + suffix
                    new_args = [self._subst_port_expr(a, loop_var, i, name_map) for a in cmd.args]
                    suffixed = Invocation(
                        name=cmd.name + suffix,
                        instance=new_instance,
                        binding=cmd.binding,
                        time_args=cmd.time_args,
                        args=new_args,
                    )
                    name_map[cmd.name] = cmd.name + suffix
                    self._process_invocation(suffixed, ir)

                elif isinstance(cmd, Pool):
                    # Pools are shared resources — keep original name, register once.
                    if cmd.name not in ir.pools:
                        self._process_pool(cmd, ir)
                    name_map[cmd.name] = cmd.name  # pool name unchanged

                elif isinstance(cmd, Connect):
                    # Substitute loop variable with concrete index before processing
                    conn = self._subst_connect(cmd, loop_var, i, name_map)
                    self._process_connect(conn, ir)

    # -------------------------------------------------------------------------
    # TimeLoop: capture as LoopRegion
    # -------------------------------------------------------------------------

    def _process_timeloop(self, loop: TimeLoop, ir: SchedulingIR):
        """Lower a TimeLoop into a LoopRegion (symbolic, not unrolled)."""
        n = self._eval_expr(loop.end)
        if n is None:
            raise ValueError(f"TimeLoop trip count must be statically known, got: {loop.end!r}")

        ii: Optional[int] = None
        if isinstance(loop.ii, Number):
            ii = loop.ii.value

        start = self._extract_timing(loop.start, ir)
        region = LoopRegion(loop_var=loop.var, start=start, ii=ii, trip_count=n)

        for cmd in loop.body:
            if isinstance(cmd, Instance):
                region.body_instances[cmd.name] = self.module_latencies.get(cmd.module, 1)

            elif isinstance(cmd, Pool):
                latency = self.module_latencies.get(cmd.module, 1)
                max_inst = None
                if cmd.size and not cmd.size.is_unbounded:
                    max_inst = self._eval_expr(cmd.size.max_count)
                region.body_pools[cmd.name] = Resource(
                    name=cmd.name,
                    module=cmd.module,
                    params=cmd.params,
                    latency=latency,
                    is_pool=True,
                    max_instances=max_inst,
                )

            elif isinstance(cmd, BundleDecl):
                region.body_bundles[cmd.name] = self._make_bundle_info(cmd)

            elif isinstance(cmd, Invocation):
                resource = cmd.instance
                # Use the global ir._op_counters so body ops get globally unique names
                # (avoids collision with flat ops or ops in other loop regions).
                if resource not in ir._op_counters:
                    ir._op_counters[resource] = 0
                op_index = ir._op_counters[resource]
                ir._op_counters[resource] += 1
                op_name = f"{resource}.{op_index}"

                timing = self._extract_timing(
                    cmd.time_args[0] if cmd.time_args else None, ir
                )
                if timing.sched_var:
                    region.sched_vars.add(timing.sched_var)
                # alias is not a free var — do not add to sched_vars

                region.body_ops[op_name] = Operation(
                    op_name=op_name,
                    resource=resource,
                    result=cmd.name,
                    timing=timing,
                    inputs=self._extract_inputs(cmd.args, ir.const_values),
                    binding=cmd.binding,
                )
                region.body_result_to_op[cmd.name] = op_name

            elif isinstance(cmd, Connect):
                dest_str = self._format_port_expr(cmd.dest)
                src_str = self._format_port_expr(cmd.src, ir.const_values)
                region.body_connects.append((dest_str, src_str))

            elif isinstance(cmd, SpaceLoop):
                self._process_nested_spaceloop(cmd, region, ir)

        # Remove aliases from sched_vars — aliases are not free variables.
        # Collect all alias names defined in this loop region.
        aliases: set = set()
        if region.start.alias:
            aliases.add(region.start.alias)
        for op in region.body_ops.values():
            if op.timing.alias:
                aliases.add(op.timing.alias)
        ir.sched_vars -= aliases
        region.sched_vars -= aliases

        ir.loops.append(region)

    def _process_nested_spaceloop(
        self, loop: SpaceLoop, region: LoopRegion, ir: SchedulingIR
    ):
        """Unroll a SpaceLoop nested inside a TimeLoop into the LoopRegion."""
        n = self._eval_expr(loop.end)
        if n is None:
            raise ValueError(f"Nested SpaceLoop trip count must be statically known, got: {loop.end!r}")

        for i in range(n):
            suffix = f"_{i}"
            name_map: Dict[str, str] = {}

            for cmd in loop.body:
                if isinstance(cmd, Instance):
                    region.body_instances[cmd.name + suffix] = \
                        self.module_latencies.get(cmd.module, 1)
                    name_map[cmd.name] = cmd.name + suffix

                elif isinstance(cmd, Pool):
                    latency = self.module_latencies.get(cmd.module, 1)
                    max_inst = None
                    if cmd.size and not cmd.size.is_unbounded:
                        max_inst = self._eval_expr(cmd.size.max_count)
                    suffixed_name = cmd.name + suffix
                    region.body_pools[suffixed_name] = Resource(
                        name=suffixed_name,
                        module=cmd.module,
                        params=cmd.params,
                        latency=latency,
                        is_pool=True,
                        max_instances=max_inst,
                    )
                    name_map[cmd.name] = suffixed_name

                elif isinstance(cmd, Invocation):
                    new_instance = name_map.get(cmd.instance, cmd.instance + suffix)
                    result_name = cmd.name + suffix
                    # Use global ir._op_counters for unique naming across all scopes
                    if new_instance not in ir._op_counters:
                        ir._op_counters[new_instance] = 0
                    op_index = ir._op_counters[new_instance]
                    ir._op_counters[new_instance] += 1
                    op_name = f"{new_instance}.{op_index}"

                    timing = self._extract_timing(
                        cmd.time_args[0] if cmd.time_args else None, ir
                    )
                    if timing.sched_var:
                        region.sched_vars.add(timing.sched_var)
                    # alias is not a free var — do not add to sched_vars

                    region.body_ops[op_name] = Operation(
                        op_name=op_name,
                        resource=new_instance,
                        result=result_name,
                        timing=timing,
                        inputs=self._extract_inputs(cmd.args, ir.const_values),
                        binding=cmd.binding,
                    )
                    region.body_result_to_op[result_name] = op_name

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _extract_timing(self, time: Optional[Time], ir: SchedulingIR) -> TimingConstraint:
        tc = TimingConstraint()
        if time is None:
            return tc
        if time.event:
            tc.base_event = time.event.name
        if time.sched_var and time.sched_var.name:
            tc.sched_var = time.sched_var.name
            ir.sched_vars.add(time.sched_var.name)
        if time.offset:
            if isinstance(time.offset, Number):
                tc.offset = time.offset.value
            elif isinstance(time.offset, RangeOffset):
                tc.range_lo = self._eval_expr(time.offset.lo) if time.offset.lo else None
                tc.range_hi = self._eval_expr(time.offset.hi) if time.offset.hi else None
            elif isinstance(time.offset, (Var, BinOp)):
                tc.offset = self._eval_expr(time.offset)
        if time.alias:
            tc.alias = time.alias
            # alias is NOT a free variable — it names an existing op's timing.
            # Do not add to sched_vars; resolved in SDC generator via alias_map.
        if time.loop_end:
            tc.loop_end = True
            # ?alias.e — sched_var holds the alias name (e.g. "tloop").
            # The actual auxiliary Z3 variable (?tloop_end) is created by the
            # SDC generator when it processes the loop region with that alias.
            # Do NOT add sched_var to ir.sched_vars here; it's not a free var.
        return tc

    def _extract_inputs(self, args, const_values: Optional[Dict[str, int]] = None) -> List[str]:
        """Extract input wire names from invocation arguments.

        Handles both PortRef (simple/qualified) and PortBundleAccess (bundle{i}).
        Bundle access is encoded as "bundle{index}" or "instance.bundle{index}".
        Const wire names are replaced with "#V" integer literals.
        """
        inputs = []
        consts = const_values or {}
        for arg in args:
            if isinstance(arg, PortRef):
                if arg.instance:
                    name = f"{arg.instance}.{arg.port}"
                else:
                    name = arg.port
                # Substitute const wire reference
                if name in consts:
                    inputs.append(f"#{consts[name]}")
                else:
                    inputs.append(name)
            elif isinstance(arg, PortBundleAccess):
                index_str = self._format_expr(arg.index)
                if arg.instance:
                    inputs.append(f"{arg.instance}.{arg.bundle}{{{index_str}}}")
                else:
                    inputs.append(f"{arg.bundle}{{{index_str}}}")
        return inputs

    def _format_port_expr(self, expr, const_values: Optional[Dict[str, int]] = None) -> str:
        """Format a port_expr AST node as a string (for body_connects encoding).

        If const_values is provided, substitutes Const wire names with "#V" literals.
        """
        consts = const_values or {}
        if isinstance(expr, PortRef):
            if expr.instance:
                name = f"{expr.instance}.{expr.port}"
            else:
                name = expr.port
            return f"#{consts[name]}" if name in consts else name
        if isinstance(expr, PortBundleAccess):
            index_str = self._format_expr(expr.index)
            if expr.instance:
                return f"{expr.instance}.{expr.bundle}{{{index_str}}}"
            return f"{expr.bundle}{{{index_str}}}"
        if isinstance(expr, Var):
            name = expr.name
            return f"#{consts[name]}" if name in consts else name
        return str(expr)

    def _eval_expr_with_var(self, expr, var: str, val: int) -> Optional[int]:
        """Evaluate expr with loop variable substituted to a concrete int."""
        if expr is None:
            return None
        if isinstance(expr, Number):
            return expr.value
        if isinstance(expr, int):
            return expr
        if isinstance(expr, Var):
            return val if expr.name == var else None
        if isinstance(expr, BinOp):
            left = self._eval_expr_with_var(expr.left, var, val)
            right = self._eval_expr_with_var(expr.right, var, val)
            if left is None or right is None:
                return None
            if expr.op == '+':
                return left + right
            if expr.op == '-':
                return left - right
            if expr.op == '*':
                return left * right
            if expr.op == '/':
                return left // right
        return None

    def _subst_port_expr(self, expr, var: str, val: int, name_map: Optional[Dict[str, str]] = None):
        """Return a new port_expr AST node with loop var substituted to concrete int.

        Also remaps instance names via name_map (for result wires like r -> r_0).
        """
        nm = name_map or {}
        if isinstance(expr, PortBundleAccess):
            new_idx = self._eval_expr_with_var(expr.index, var, val)
            new_instance = nm.get(expr.instance, expr.instance) if expr.instance else None
            return PortBundleAccess(
                instance=new_instance,
                bundle=expr.bundle,
                index=Number(value=new_idx) if new_idx is not None else expr.index,
            )
        if isinstance(expr, PortRef):
            new_instance = nm.get(expr.instance, expr.instance) if expr.instance else None
            return PortRef(instance=new_instance, port=expr.port)
        if isinstance(expr, Var):
            return Var(name=nm.get(expr.name, expr.name))
        return expr

    def _subst_connect(self, conn, var: str, val: int, name_map: Dict[str, str]):
        """Return a Connect with loop var substituted and result-wire names remapped."""
        from ast_nodes import Connect
        dest = self._subst_port_expr(conn.dest, var, val, name_map)
        src = self._subst_port_expr(conn.src, var, val, name_map)
        return Connect(dest=dest, src=src)

    def _format_expr(self, expr) -> str:
        """Format an expression as a string (for bundle index encoding)."""
        if isinstance(expr, Number):
            return str(expr.value)
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, BinOp):
            return f"{self._format_expr(expr.left)}{expr.op}{self._format_expr(expr.right)}"
        return "?"

    def _eval_expr(self, expr) -> Optional[int]:
        if expr is None:
            return None
        if isinstance(expr, Number):
            return expr.value
        if isinstance(expr, int):
            return expr
        return None


def build_ir(component: Component, module_latencies: Optional[Dict[str, int]] = None) -> SchedulingIR:
    """Build IR from a Component AST node."""
    return IRBuilder(module_latencies).build(component)
