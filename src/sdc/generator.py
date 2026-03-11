"""
SDC Constraint Generator

Generates the SDCModel from a SchedulingIR.

Constraint categories generated:
  1. Non-negative:       op.t >= 0, ?var >= 0
  2. Sched-var links:    op.t == ?var + offset  (or range)
  3. Input windows:      op.t in [input_start, input_end)
  4. Op-to-op:           consumer.t >= producer.t + latency
  5. Output deadlines:   producer.t + latency <= deadline
  6. Timing:             'G+2 -> op.t == 2; 'G+[4..6] -> 4 <= op.t <= 6
  7. Resource:           at most N pool operations active at once
  8. Binding:            named/explicit pool instance assignment
  9. Non-overlap:        pairwise disjunctive constraint for same-pool ops

  Loop region constraints (per LoopRegion in ir.loops):
  10. Intra-iteration:   consumer.t >= producer.t + latency (within one body)
  11. Recurrence:        II >= ceil(latency/distance) for loop-carried edges
  12. MRT:               II >= ceil(k * lat / N) per resource
  13. II non-negative:   II >= 1

Alias resolution:
  When a TimingConstraint carries an alias (from ?name=<time>), the alias
  name is mapped to the same existing timing/sched variable rather than
  creating a new one.  Look-ups that fail the primary map fall through to
  the alias map automatically.
"""

import math
from typing import Dict, List, Optional, Set, Tuple

from ir.nodes import SchedulingIR, Operation, Resource, LoopRegion
from sdc.model import SDCModel, Constraint, ConstraintType


class SDCGenerator:
    """Generates SDC constraints from Scheduling IR."""

    def __init__(self, ir: SchedulingIR):
        self.ir = ir
        self.model = SDCModel(
            component_name=ir.name,
            resources=ir.pools.copy(),
            operations=ir.operations.copy(),
            latencies={
                **ir.module_latencies,
                **{p: r.latency for p, r in ir.pools.items()},
            },
        )
        # alias_name -> op timing var it names, or None for event-anchored aliases
        # e.g. "ta" -> "A.0.t",  "tloop" -> None (loop start = event 'G, offset 0)
        self._alias_map: Dict[str, Optional[str]] = {}
        # reverse: op_timing_var -> alias_name, e.g. "A.0.t" -> "ta"
        self._reverse_alias: Dict[str, str] = {}
        # loop-end aliases: alias_name -> end_var, e.g. "tloop" -> "?tloop_end"
        self._loop_end_alias_map: Dict[str, str] = {}

    def generate(self) -> SDCModel:
        self._collect_variables()
        self.model.tvar_aliases = dict(self._reverse_alias)
        self._gen_non_negative()
        self._gen_sched_var_links()
        self._gen_input_constraints()
        self._gen_op_to_op_constraints()
        self._gen_output_deadlines()
        self._gen_timing_constraints()
        self._gen_resource_constraints()
        self._gen_binding_constraints()
        self._gen_non_overlap_constraints()
        for idx, region in enumerate(self.ir.loops):
            self._gen_loop_region_constraints(region, idx)
        return self.model

    # ------------------------------------------------------------------
    # Variable names
    # ------------------------------------------------------------------

    def _timing_var(self, op_name: str) -> str:
        return f"{op_name}.t"

    def _binding_var(self, op_name: str) -> str:
        return f"{op_name}.i"

    def _sched_var(self, name: str) -> str:
        return f"?{name}"

    def _ii_var(self, region_idx: int) -> str:
        return f"Loop.{region_idx}.II"

    # ------------------------------------------------------------------
    # Variable collection
    # ------------------------------------------------------------------

    def _collect_variables(self):
        # First pass: register aliases from all ops (flat + loop body)
        # alias_name -> op_timing_var of the op that defines the alias
        self._build_alias_map()

        # Register loop-end aux variables as free sched vars
        for end_var in self._loop_end_alias_map.values():
            self.model.sched_vars.add(end_var)

        for op_name, op in self.ir.operations.items():
            self.model.timing_vars.add(self._timing_var(op_name))
            if op.resource in self.ir.pools:
                self.model.binding_vars.add(self._binding_var(op_name))
            sv = op.timing.sched_var
            if sv:
                if op.timing.loop_end:
                    # The actual free variable is ?alias_end — track for constraint gen
                    self.model.sched_var_to_ops.setdefault(sv, []).append(op_name)
                elif sv not in self._alias_map:
                    # Free sched var — add as Z3 variable
                    self.model.sched_vars.add(self._sched_var(sv))
                    # Track for constraint generation
                    self.model.sched_var_to_ops.setdefault(sv, []).append(op_name)
                else:
                    # Alias reference — track for constraint generation
                    self.model.sched_var_to_ops.setdefault(sv, []).append(op_name)

    def _build_alias_map(self):
        """
        Build alias_name -> op_timing_var for every op that carries an alias.

        ?ta=?tloop+[3..] on op A.0 means: 'ta' is a name for A.0.t.
        ?tloop='G on the loop start means: 'tloop' is a name for the loop anchor (offset 0).

        Aliases are NOT free variables; they are display names for existing timing vars.
        When another op references sched_var='ta', we substitute A.0.t for ?ta.
        Also builds the reverse map op_timing_var -> alias_name for display annotation.

        For loop-end refs (?alias.e): the IR builder records alias -> end_var_name in
        region.loop_end_vars (e.g. "tloop" -> "?tloop_end").  We map the alias name to
        the end-var string so that _resolve_sched_var("tloop") when loop_end=True returns
        "?tloop_end".  We use a separate _loop_end_alias_map for this distinction.
        """
        # Flat ops
        for op_name, op in self.ir.operations.items():
            if op.timing.alias:
                tvar = self._timing_var(op_name)
                self._alias_map[op.timing.alias] = tvar
                self._reverse_alias[tvar] = op.timing.alias
        # Loop body ops
        for idx, region in enumerate(self.ir.loops):
            # Loop start alias (e.g. ?tloop='G): introduce Loop.N.t as a timing var
            # and map the alias to it, just like ?ta maps to A.0.t.
            loop_tvar = f"Loop.{idx}.t"
            if region.start.alias:
                self._alias_map[region.start.alias] = loop_tvar
                self._reverse_alias[loop_tvar] = region.start.alias
            for op_name, op in region.body_ops.items():
                if op.timing.alias:
                    tvar = self._timing_var(op_name)
                    self._alias_map[op.timing.alias] = tvar
                    self._reverse_alias[tvar] = op.timing.alias
            # Loop-end aliases: alias -> ?alias_end (auxiliary free variable)
            for alias, end_var in region.loop_end_vars.items():
                self._loop_end_alias_map[alias] = end_var

    def _resolve_sched_var(self, sv_name: str) -> Optional[str]:
        """
        Resolve a sched_var name to its constraint variable.
        - Alias pointing to an op: returns that op's timing var (e.g. "A.0.t")
        - Alias pointing to a loop-start event (None): returns None (constant 0)
        - Free sched var: returns "?sv_name"
        """
        if sv_name in self._alias_map:
            return self._alias_map[sv_name]
        return self._sched_var(sv_name)

    def _alias_display(self, sv_name: str) -> str:
        """
        Human-readable display for a sched var reference.
        - Free var: "?ta"
        - Alias to op: "A.0.t(?ta)"
        - Alias to event: "'G(?tloop)"
        """
        if sv_name in self._alias_map:
            target = self._alias_map[sv_name]
            if target is None:
                return f"'G(?{sv_name})"
            return f"{target}(?{sv_name})"
        return f"?{sv_name}"

    def _annotate_tvar(self, tvar: str) -> str:
        """
        Annotate a timing variable with its alias if one exists.
        e.g. "A.0.t" -> "A.0.t(?ta)", "M.0.t" -> "M.0.t"
        """
        alias = self._reverse_alias.get(tvar)
        if alias:
            return f"{tvar}(?{alias})"
        return tvar

    def _get_latency(self, resource_name: str) -> int:
        return self.model.latencies.get(resource_name, 1)

    # ------------------------------------------------------------------
    # Constraint generators (flat operations — unchanged from before)
    # ------------------------------------------------------------------

    def _gen_non_negative(self):
        for var in sorted(self.model.timing_vars | self.model.sched_vars):
            self.model.add(Constraint(
                kind=ConstraintType.NON_NEGATIVE,
                lhs=var, rhs=None, constant=0,
                is_lower_bound=True, reason=f"{var} >= 0",
            ))

    def _gen_sched_var_links(self):
        """op.t == rhs + offset, where rhs is a free sched var or an aliased op.t."""
        for var_name, op_names in self.model.sched_var_to_ops.items():
            for op_name in op_names:
                op = self.ir.operations[op_name]
                op_var = self._timing_var(op_name)
                timing = op.timing

                # loop_end ops: link to ?alias_end auxiliary var
                if timing.loop_end:
                    end_var = self._loop_end_alias_map.get(var_name)
                    if end_var is None:
                        continue
                    rhs_var = end_var
                    display_name = f"?{var_name}.e"
                else:
                    rhs_var = self._resolve_sched_var(var_name)
                    display_name = self._alias_display(var_name)

                has_range = timing.range_lo is not None or timing.range_hi is not None

                if has_range:
                    lo = timing.range_lo if timing.range_lo is not None else 0
                    self.model.add(Constraint(
                        kind=ConstraintType.SCHED_VAR_LINK,
                        lhs=op_var, rhs=rhs_var, constant=lo,
                        is_lower_bound=True, is_equality=False,
                        reason=f"{display_name}+[{timing.range_lo if timing.range_lo is not None else ''}..]",
                    ))
                    if timing.range_hi is not None:
                        self.model.add(Constraint(
                            kind=ConstraintType.SCHED_VAR_LINK,
                            lhs=op_var, rhs=rhs_var, constant=timing.range_hi,
                            is_lower_bound=False, is_equality=False,
                            reason=f"{display_name}+[..{timing.range_hi}]",
                        ))
                else:
                    self.model.add(Constraint(
                        kind=ConstraintType.SCHED_VAR_LINK,
                        lhs=op_var, rhs=rhs_var, constant=timing.offset or 0,
                        is_equality=True,
                        reason=f"{display_name}+{timing.offset}" if timing.offset else display_name,
                    ))

    def _gen_input_constraints(self):
        """op.t in [input_start, input_end) — skip end constraint if end is anonymous (?)."""
        input_windows: Dict[str, Tuple[int, Optional[int]]] = {}
        for port_name, port in self.ir.inputs.items():
            start_time = (port.start_offset if port.start_offset is not None else 0) \
                if port.start_event else None
            # Only set end_time if it is a concrete event (not anonymous ?)
            end_time = None
            if port.end_event and port.end_sched_var is None:
                end_time = port.end_offset if port.end_offset is not None else 0
            if start_time is not None:
                input_windows[port_name] = (start_time, end_time)

        for op_name, op in self.ir.operations.items():
            op_var = self._timing_var(op_name)
            for inp in op.inputs:
                base_name = inp.split('.')[0].split('{')[0]
                if base_name not in input_windows:
                    continue
                start_time, end_time = input_windows[base_name]
                self.model.add(Constraint(
                    kind=ConstraintType.INPUT_WINDOW,
                    lhs=op_var, rhs=None, constant=start_time,
                    is_lower_bound=True,
                    reason=f"[input: {base_name}, avail@{start_time}]",
                ))
                if end_time is not None and end_time > start_time:
                    self.model.add(Constraint(
                        kind=ConstraintType.INPUT_WINDOW,
                        lhs=op_var, rhs=None, constant=end_time - 1,
                        is_lower_bound=False,
                        reason=f"[input: {base_name}, expires@{end_time}]",
                    ))

    def _gen_op_to_op_constraints(self):
        """consumer.t >= producer.t + latency."""
        producers: Dict[str, Tuple[str, int, str]] = {}
        for op_name, op in self.ir.operations.items():
            latency = self._get_latency(op.resource)
            result = op.result
            producers[f"{result}.out"] = (op_name, latency, op.resource)
            producers[result] = (op_name, latency, op.resource)

        for op_name, op in self.ir.operations.items():
            consumer_var = self._timing_var(op_name)
            for inp in op.inputs:
                if inp not in producers:
                    continue
                producer_op, latency, resource = producers[inp]
                if producer_op == op_name:
                    continue
                self.model.add(Constraint(
                    kind=ConstraintType.OP_TO_OP,
                    lhs=consumer_var, rhs=self._timing_var(producer_op),
                    constant=latency, is_lower_bound=True,
                    reason=f"{self._annotate_tvar(consumer_var)} <- {self._annotate_tvar(self._timing_var(producer_op))} [latency({resource})={latency}]",
                ))

    def _gen_output_deadlines(self):
        """producer.t + latency <= deadline."""
        for out_name, out_port in self.ir.outputs.items():
            if out_name not in self.ir.output_connections:
                continue
            producer_op, _ = self.ir.output_connections[out_name]
            if producer_op not in self.ir.operations:
                continue
            op = self.ir.operations[producer_op]
            latency = self._get_latency(op.resource)
            if out_port.start_event and out_port.start_sched_var is None:
                deadline = out_port.start_offset if out_port.start_offset is not None else 0
                self.model.add(Constraint(
                    kind=ConstraintType.OUTPUT_DEADLINE,
                    lhs=self._timing_var(producer_op), rhs=None,
                    constant=deadline - latency, is_lower_bound=False,
                    reason=(f"{producer_op} -> {out_name} "
                            f"[deadline='{out_port.start_event}+{deadline}, "
                            f"latency({op.resource})={latency}]"),
                ))

    def _gen_timing_constraints(self):
        """'G+2 -> op.t == 2; 'G+[4..6] -> 4 <= op.t <= 6."""
        for op_name, op in self.ir.operations.items():
            op_var = self._timing_var(op_name)
            timing = op.timing

            if timing.offset is not None and timing.base_event:
                self.model.add(Constraint(
                    kind=ConstraintType.TIMING_EQUALITY,
                    lhs=op_var, rhs=None, constant=timing.offset, is_equality=True,
                    reason=f"'{timing.base_event}+{timing.offset}",
                ))

            if timing.sched_var is None:
                if timing.range_lo is not None:
                    self.model.add(Constraint(
                        kind=ConstraintType.TIMING_LOWER,
                        lhs=op_var, rhs=None, constant=timing.range_lo,
                        is_lower_bound=True, reason=f">= {timing.range_lo}",
                    ))
                if timing.range_hi is not None:
                    self.model.add(Constraint(
                        kind=ConstraintType.TIMING_UPPER,
                        lhs=op_var, rhs=None, constant=timing.range_hi,
                        is_lower_bound=False, reason=f"<= {timing.range_hi}",
                    ))

    def _gen_resource_constraints(self):
        """At most max_instances pool operations active at once (None = solver determines)."""
        for pool_name, pool in self.ir.pools.items():
            ops = [n for n, op in self.ir.operations.items()
                   if op.resource == pool_name]
            if not ops:
                continue
            self.model.add(Constraint(
                kind=ConstraintType.RESOURCE,
                resource=pool_name, operations=ops,
                max_concurrent=pool.max_instances, latency=pool.latency,
                reason=f"pool has {pool.max_instances if pool.max_instances is not None else '?'} units",
            ))

    def _gen_binding_constraints(self):
        """Named/explicit pool instance assignment."""
        binding_groups: Dict[str, List[str]] = {}
        for op_name, op in self.ir.operations.items():
            if not op.binding:
                continue
            if op.binding.kind == 'named':
                binding_groups.setdefault(op.binding.value, []).append(op_name)
            elif op.binding.kind == 'explicit':
                from ast_nodes import Number
                index_value = op.binding.value
                if isinstance(index_value, Number):
                    index_value = index_value.value
                elif not isinstance(index_value, int):
                    index_value = int(index_value) if index_value is not None else 0
                self.model.add(Constraint(
                    kind=ConstraintType.BINDING_EXPLICIT,
                    lhs=self._binding_var(op_name), rhs=None,
                    constant=index_value, is_equality=True,
                    reason=f"explicit binding [{index_value}]",
                ))

        for bind_var, ops in binding_groups.items():
            if len(ops) > 1:
                self.model.add(Constraint(
                    kind=ConstraintType.BINDING_EQUALITY,
                    binding_var=bind_var, operations=ops,
                    reason=f"ops share binding ?{bind_var}",
                ))

    def _gen_non_overlap_constraints(self):
        """Pairwise disjunctive non-overlap for operations on the same pool.

        Only generated when ops might share a unit (len(ops) > pool.max_instances).
        When each op has its own dedicated unit, no non-overlap is needed.
        """
        for pool_name, pool in self.ir.pools.items():
            if pool.max_instances is None:
                continue
            ops = [n for n, op in self.ir.operations.items()
                   if op.resource == pool_name]
            if len(ops) <= pool.max_instances:
                continue  # Each op gets its own unit — no conflict possible
            for i, op1 in enumerate(ops):
                for op2 in ops[i+1:]:
                    self.model.add(Constraint(
                        kind=ConstraintType.NON_OVERLAP,
                        operations=[op1, op2],
                        latency=pool.latency,
                        resource=pool_name,
                        reason=f"same pool {pool_name}",
                    ))

    # ------------------------------------------------------------------
    # Loop region constraints
    # ------------------------------------------------------------------

    def _gen_loop_region_constraints(self, region: LoopRegion, idx: int):
        """
        Generate all modulo-scheduling constraints for one LoopRegion.

        (a) Intra-iteration data dependencies: within one canonical body.
        (b) Recurrence constraints: II >= ceil(latency / distance) for each
            loop-carried edge (bundle wire chain: s{i+1} = producer.out).
        (c) MRT lower bounds: II >= ceil(k * lat / N) per resource pool.
        (d) II non-negative / minimum.

        Alias resolution: if the loop start TimingConstraint has an alias
        (e.g. ?tloop='G), the alias name is already recorded in _alias_map
        during _collect_variables; body op timing vars relative to the loop
        start are expressed as offsets from 0 (all times in [0, II)).
        """
        ii_var = self._ii_var(idx)
        self.model.loop_regions.append(region)
        self.model.loop_ii_vars[len(self.model.loop_regions) - 1] = ii_var

        # Register Loop.N.t timing variable and fix it to the loop start offset
        loop_tvar = f"Loop.{idx}.t"
        self.model.timing_vars.add(loop_tvar)
        start_offset = region.start.offset or 0
        self.model.add(Constraint(
            kind=ConstraintType.TIMING_EQUALITY,
            lhs=loop_tvar, rhs=None, constant=start_offset, is_equality=True,
            reason=f"loop start '{region.start.base_event or self.ir.start_event}+{start_offset}",
        ))

        # Collect resources used in this loop body (merge body_pools + outer pools)
        all_pools: Dict[str, Resource] = {**self.ir.pools, **region.body_pools}
        all_latencies: Dict[str, int] = {
            **self.model.latencies,
            **{p: r.latency for p, r in region.body_pools.items()},
        }

        # Register timing variables for body ops and add them to model.operations
        # so the solver can look up latencies for makespan computation.
        for op_name, op in region.body_ops.items():
            self.model.timing_vars.add(self._timing_var(op_name))
            if op.resource in all_pools:
                self.model.binding_vars.add(self._binding_var(op_name))
            self.model.operations[op_name] = op
            # Ensure latency is in model.latencies
            if op.resource not in self.model.latencies:
                self.model.latencies[op.resource] = all_latencies.get(op.resource, 1)

        # II is handled by the solver's iteration loop (not a Z3 variable).
        # ii_var is kept for display annotation only.

        # Compute sink ops: body ops whose result is not consumed by any other body op.
        body_consumed: set = set()
        for op in region.body_ops.values():
            for inp in op.inputs:
                body_consumed.add(inp.split('.')[0].split('{')[0])
                body_consumed.add(inp)
        sink_ops = [
            (op_name, op)
            for op_name, op in region.body_ops.items()
            if op.result not in body_consumed and f"{op.result}.out" not in body_consumed
        ]

        # Loop-end lower bounds: ?alias_end >= sink_op.t + lat + II*(trip_count-1)
        # This correctly encodes: loop ends when the last iteration's last op finishes.
        for end_var in region.loop_end_vars.values():
            ii_display = str(region.ii) if region.ii is not None else ii_var
            for sink_op_name, sink_op in sink_ops:
                lat = all_latencies.get(sink_op.resource, 1)
                body_tvar = self._timing_var(sink_op_name)
                self.model.add(Constraint(
                    kind=ConstraintType.LOOP_END_SINK,
                    lhs=end_var,
                    rhs=loop_tvar,           # Loop.N.t — absolute loop start
                    body_tvar=body_tvar,
                    body_latency=lat,
                    ii_var=ii_var if region.ii is None else None,
                    trip_count=region.trip_count,
                    constant=region.ii if region.ii is not None else 0,
                    is_lower_bound=True,
                    reason=f"{end_var} >= {loop_tvar} + ({body_tvar} - {loop_tvar}) + {lat} + {ii_display}*{region.trip_count-1} (sink)",
                ))

        # (a) Intra-iteration data dependencies
        # Build producer map from body op results
        body_producers: Dict[str, Tuple[str, int]] = {}
        for op_name, op in region.body_ops.items():
            lat = all_latencies.get(op.resource, 1)
            body_producers[op.result] = (op_name, lat)
            body_producers[f"{op.result}.out"] = (op_name, lat)

        for op_name, op in region.body_ops.items():
            consumer_var = self._timing_var(op_name)
            for inp in op.inputs:
                # Strip bundle index for lookup: "s{i}" -> "s"
                base = inp.split('.')[0].split('{')[0]
                key = inp if inp in body_producers else (f"{base}.out" if f"{base}.out" in body_producers else base)
                if key not in body_producers:
                    continue
                prod_op, lat = body_producers[key]
                if prod_op == op_name:
                    continue
                self.model.add(Constraint(
                    kind=ConstraintType.LOOP_INTRA_DEP,
                    lhs=consumer_var,
                    rhs=self._timing_var(prod_op),
                    constant=lat,
                    is_lower_bound=True,
                    reason=f"{self._annotate_tvar(consumer_var)} <- {self._annotate_tvar(self._timing_var(prod_op))} [lat={lat}]",
                ))

        # (b) Recurrence constraints from loop-carried edges.
        # Each body_connect "s{i+1} = producer.out" is an inter-iteration dependence
        # with distance=1: consumer in iteration k+1 reads producer from iteration k.
        # Encoded as: producer.t - consumer.t <= II*dist - lat
        # (su - sv <= II*dist - lat, standard SDC modulo form from the paper)
        # With II as a constant substituted at solve time, this is a pure difference constraint.
        rec_mii = 1  # RecMII: minimum II from recurrences
        for dest_str, src_str in region.body_connects:
            src_base = src_str.split('.')[0]
            producer_info = body_producers.get(src_str) or body_producers.get(f"{src_base}.out")
            if producer_info is None:
                continue
            prod_op, lat = producer_info
            # Find the consumer op that reads dest_str (the carried wire)
            dest_base = dest_str.split('{')[0]
            consumer_ops = [
                (n, op) for n, op in region.body_ops.items()
                if any(dest_base in inp or dest_str in inp for inp in op.inputs)
            ]
            distance = 1
            ii_lower = math.ceil(lat / distance)
            rec_mii = max(rec_mii, ii_lower)
            # Emit inter-iteration SDC edge: prod_op.t - consumer.t <= II*dist - lat
            # Stored with ii_dist so solver can substitute fixed II.
            for cons_op_name, _ in consumer_ops:
                self.model.add(Constraint(
                    kind=ConstraintType.LOOP_RECURRENCE,
                    lhs=self._timing_var(prod_op),
                    rhs=self._timing_var(cons_op_name),
                    constant=-lat,   # will become II*dist - lat at solve time
                    ii_dist=distance,
                    is_lower_bound=False,  # upper bound: prod.t <= cons.t + II*dist - lat
                    reason=f"inter-iter: {prod_op}→{dest_str}→{cons_op_name}(next i): "
                           f"prod.t - cons.t <= II*{distance} - {lat}",
                ))

        # (c) MRT: compute ResII = max over resources of ceil(k*lat/N)
        res_mii = 1
        resource_use_count: Dict[str, int] = {}
        for op in region.body_ops.values():
            if op.resource in all_pools:
                resource_use_count[op.resource] = resource_use_count.get(op.resource, 0) + 1

        for res_name, k in resource_use_count.items():
            pool = all_pools.get(res_name)
            if pool is None or pool.max_instances is None:
                continue
            lat = all_latencies.get(res_name, 1)
            N = pool.max_instances
            # MRT: ceil(k*lat/N) for resource pressure across iterations.
            # Also: each op fires every II cycles on its unit => II >= lat (self-overlap).
            ii_lower = max(math.ceil(k * lat / N), lat)
            res_mii = max(res_mii, ii_lower)
            # MRT is enforced by modulo non-overlap constraints (not a separate constraint)

        mii = max(rec_mii, res_mii) if region.ii is None else region.ii

        # Build body op pairs per pool for modulo non-overlap in the solver
        body_op_pairs: Dict[str, List] = {}
        body_self_overlap: Dict[str, int] = {}
        for pool_name, pool in all_pools.items():
            ops = [n for n, op in region.body_ops.items() if op.resource == pool_name]
            if not ops:
                continue
            lat = all_latencies.get(pool_name, 1)
            # Any pool with ops needs self-overlap: each op repeats every II cycles => II >= lat
            body_self_overlap[pool_name] = lat
            if len(ops) < 2:
                continue
            pairs = []
            for i, op1 in enumerate(ops):
                for op2 in ops[i+1:]:
                    pairs.append((op1, op2, lat))
            body_op_pairs[pool_name] = pairs

        from sdc.model import LoopModuloInfo
        self.model.loop_modulo_infos.append(LoopModuloInfo(
            region_idx=idx,
            mii=mii,
            ii_var=ii_var,
            body_op_pairs=body_op_pairs,
            body_self_overlap=body_self_overlap,
        ))

        # (d) Intra-body sched var links — aliases resolve to op timing vars, not free vars
        body_sv_to_ops: Dict[str, List[str]] = {}
        for op_name, op in region.body_ops.items():
            sv = op.timing.sched_var
            if sv and sv not in self._alias_map:
                # Only add as free sched var if not an alias
                self.model.sched_vars.add(self._sched_var(sv))
                body_sv_to_ops.setdefault(sv, []).append(op_name)
            elif sv and sv in self._alias_map:
                # It's an alias reference — generate constraint against aliased op.t
                body_sv_to_ops.setdefault(sv, []).append(op_name)

        for sv_name, op_names in body_sv_to_ops.items():
            rhs_var = self._resolve_sched_var(sv_name)
            display_name = self._alias_display(sv_name)
            for op_name in op_names:
                op = region.body_ops[op_name]
                op_var = self._timing_var(op_name)
                timing = op.timing
                has_range = timing.range_lo is not None or timing.range_hi is not None
                if has_range:
                    lo = timing.range_lo if timing.range_lo is not None else 0
                    self.model.add(Constraint(
                        kind=ConstraintType.SCHED_VAR_LINK,
                        lhs=op_var, rhs=rhs_var, constant=lo,
                        is_lower_bound=True, is_equality=False,
                        reason=f"body: {display_name}+[{lo}..]",
                    ))
                    if timing.range_hi is not None:
                        self.model.add(Constraint(
                            kind=ConstraintType.SCHED_VAR_LINK,
                            lhs=op_var, rhs=rhs_var, constant=timing.range_hi,
                            is_lower_bound=False, is_equality=False,
                            reason=f"body: {display_name}+[..{timing.range_hi}]",
                        ))
                else:
                    self.model.add(Constraint(
                        kind=ConstraintType.SCHED_VAR_LINK,
                        lhs=op_var, rhs=rhs_var, constant=timing.offset or 0,
                        is_equality=True,
                        reason=f"body: {display_name}" + (f"+{timing.offset}" if timing.offset else ""),
                    ))

        # Body non-negative timing vars
        for op_name in region.body_ops:
            op_var = self._timing_var(op_name)
            self.model.add(Constraint(
                kind=ConstraintType.NON_NEGATIVE,
                lhs=op_var, rhs=None, constant=0,
                is_lower_bound=True, reason=f"{op_var} >= 0",
            ))

        # Body resource / binding / non-overlap (same as flat, scoped to body)
        self._gen_body_resource_constraints(region, all_pools, all_latencies)
        self._gen_body_binding_constraints(region)
        self._gen_body_non_overlap_constraints(region, all_pools)

    def _gen_body_resource_constraints(
        self, region: LoopRegion,
        all_pools: Dict[str, Resource],
        all_latencies: Dict[str, int],
    ):
        for pool_name, pool in all_pools.items():
            ops = [n for n, op in region.body_ops.items() if op.resource == pool_name]
            if not ops:
                continue
            self.model.add(Constraint(
                kind=ConstraintType.RESOURCE,
                resource=pool_name, operations=ops,
                max_concurrent=pool.max_instances, latency=pool.latency,
                reason=f"body pool {pool_name} has {pool.max_instances if pool.max_instances is not None else '?'} units",
            ))

    def _gen_body_binding_constraints(self, region: LoopRegion):
        binding_groups: Dict[str, List[str]] = {}
        for op_name, op in region.body_ops.items():
            if not op.binding:
                continue
            if op.binding.kind == 'named':
                binding_groups.setdefault(op.binding.value, []).append(op_name)
            elif op.binding.kind == 'explicit':
                from ast_nodes import Number
                val = op.binding.value
                if isinstance(val, Number):
                    val = val.value
                elif not isinstance(val, int):
                    val = int(val) if val is not None else 0
                self.model.add(Constraint(
                    kind=ConstraintType.BINDING_EXPLICIT,
                    lhs=self._binding_var(op_name), rhs=None,
                    constant=val, is_equality=True,
                    reason=f"body explicit binding [{val}]",
                ))
        for bind_var, ops in binding_groups.items():
            if len(ops) > 1:
                self.model.add(Constraint(
                    kind=ConstraintType.BINDING_EQUALITY,
                    binding_var=bind_var, operations=ops,
                    reason=f"body ops share binding ?{bind_var}",
                ))

    def _gen_body_non_overlap_constraints(
        self, region: LoopRegion, all_pools: Dict[str, Resource]
    ):
        """Only generate non-overlap when ops might share a unit (more ops than units)."""
        for pool_name, pool in all_pools.items():
            if pool.max_instances is None:
                continue
            ops = [n for n, op in region.body_ops.items() if op.resource == pool_name]
            if len(ops) <= pool.max_instances:
                continue  # Each op gets its own unit — no conflict possible
            for i, op1 in enumerate(ops):
                for op2 in ops[i+1:]:
                    self.model.add(Constraint(
                        kind=ConstraintType.NON_OVERLAP,
                        operations=[op1, op2],
                        latency=pool.latency,
                        resource=pool_name,
                        reason=f"body same pool {pool_name}",
                    ))


def generate_sdc(ir: SchedulingIR) -> SDCModel:
    """Generate SDC model from IR."""
    return SDCGenerator(ir).generate()
