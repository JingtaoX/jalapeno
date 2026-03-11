"""
SDC Solver using Z3

Solves the constraint model to find a valid schedule.

For flat (non-loop) components:
  Single Z3 solve with all constraints.

For loop regions (modulo scheduling):
  II iteration loop: start at MII (analytical lower bound), fix II as a
  constant, solve Z3, if UNSAT increment II and retry.

  With II fixed, all modulo constraints become linear:
  - Inter-iteration recurrence edges: prod.t - cons.t <= II*dist - lat
    (pure difference constraint once II is a constant)
  - Modulo non-overlap: for each pair of body ops on the same pool unit,
    use Z3's native % operator with II as a constant:
      gap = (tb - ta) % II        # forward circular distance, in [0, II)
      gap >= lat                  # b is at least lat after a in the ring
      II - gap >= lat             # a is at least lat after b in the ring (wrap-around)
    Both constraints together enforce min(gap, II-gap) >= lat.

Optimization goal: minimize makespan (max op end time).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

try:
    from z3 import (
        Int, Solver, Optimize, sat, unsat, unknown,
        And, Or, Not, If, Sum, Implies,
        IntVal,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: z3-solver not installed. Run: pip install z3-solver")

from sdc import SDCModel, Constraint, ConstraintType
from sdc.model import LoopModuloInfo


class SolveStatus(Enum):
    """Result status of solving."""
    OPTIMAL = "optimal"
    SATISFIABLE = "satisfiable"
    UNSATISFIABLE = "unsatisfiable"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class Schedule:
    """
    Result of scheduling: assignment of times to operations.
    """
    status: SolveStatus
    component_name: str = ""

    # Variable assignments: var_name -> time
    assignments: Dict[str, int] = field(default_factory=dict)

    # Derived info
    makespan: Optional[int] = None        # Total latency (max end time)
    start_time: Optional[int] = None      # Min start time

    # Resource assignments (if solved)
    # op_name -> resource_instance_index
    resource_assignments: Dict[str, int] = field(default_factory=dict)

    # Pool latencies: pool_name -> latency
    pool_latencies: Dict[str, int] = field(default_factory=dict)

    # Error/debug info
    message: str = ""

    # Display annotation: op_timing_var -> alias_name, e.g. "A.0.t" -> "ta"
    tvar_aliases: Dict[str, str] = field(default_factory=dict)

    def print_schedule(self):
        """Print the schedule in a readable format."""
        print(f"\nSchedule: {self.component_name}")

        if self.status in (SolveStatus.OPTIMAL, SolveStatus.SATISFIABLE):
            print(f"  Makespan: {self.makespan} cycles")

            # Separate timing vars (.t and ?vars) from binding vars (.i)
            timing_vars = {}
            binding_vars = {}
            for var, val in self.assignments.items():
                if var.endswith('.i'):
                    binding_vars[var] = val
                else:
                    timing_vars[var] = val

            # Print timing assignments
            print(f"\n  Timing assignments:")
            sorted_timing = sorted(timing_vars.items(), key=lambda x: x[1])
            for var, time in sorted_timing:
                alias = self.tvar_aliases.get(var)
                label = f"{var}(?{alias})" if alias else var
                print(f"    {label} = {time}")

            # Print binding assignments (if any)
            if binding_vars:
                print(f"\n  Binding assignments:")
                sorted_binding = sorted(binding_vars.items(), key=lambda x: x[0])
                for var, unit in sorted_binding:
                    print(f"    {var} = {unit}")

            # Print non-overlap verification (operations grouped by unit)
            if self.resource_assignments:
                self._print_non_overlap_verification(timing_vars)
        else:
            print(f"  Message: {self.message}")

    def _print_non_overlap_verification(self, timing_vars: Dict[str, int]):
        """
        Print non-overlap verification showing that operations on the same unit
        don't overlap in time.
        """
        pool_units: Dict[str, Dict[int, List[Tuple[str, int]]]] = {}

        for op_name, unit in self.resource_assignments.items():
            pool_name = op_name.rsplit('.', 1)[0] if '.' in op_name else op_name

            if pool_name not in pool_units:
                pool_units[pool_name] = {}
            if unit not in pool_units[pool_name]:
                pool_units[pool_name][unit] = []

            timing_var = f"{op_name}.t"
            start_time = timing_vars.get(timing_var, 0)
            pool_units[pool_name][unit].append((op_name, start_time))

        print(f"\n  Non-overlap verification:")
        for pool_name in sorted(pool_units.keys()):
            units = pool_units[pool_name]
            latency = self.pool_latencies.get(pool_name, 1)
            for unit in sorted(units.keys()):
                ops = sorted(units[unit], key=lambda x: x[1])
                if len(ops) > 1:
                    intervals = []
                    for op_name, start in ops:
                        intervals.append(f"{op_name}@[{start},{start+latency})")
                    print(f"    {pool_name}[{unit}]: {' -> '.join(intervals)}")
                else:
                    op_name, start = ops[0]
                    print(f"    {pool_name}[{unit}]: {op_name}@[{start},{start+latency})")


class Z3Solver:
    """
    Z3-based solver for SDC constraints.

    Takes a fixed II value for each loop region (or None if region.ii is set).
    The II iteration loop is managed externally by solve_sdc().
    """

    def __init__(self, model: SDCModel, loop_iis: Dict[int, int], optimize: bool = True):
        """
        Args:
            model: SDC constraint model
            loop_iis: region_idx -> fixed II value for this solve attempt
            optimize: If True, minimize makespan.
        """
        if not Z3_AVAILABLE:
            raise RuntimeError("z3-solver not installed")

        self.model = model
        self.loop_iis = loop_iis  # region_idx -> II constant
        self.optimize = optimize

        self.z3_vars: Dict[str, any] = {}
        self.z3_resource_vars: Dict[str, Dict[str, any]] = {}

        if optimize:
            self.solver = Optimize()
        else:
            self.solver = Solver()

    def solve(self) -> Schedule:
        """Solve the constraint model."""
        try:
            self._create_variables()
            self._add_difference_constraints()
            self._add_resource_constraints()
            self._add_binding_constraints()
            self._add_non_overlap_constraints()
            self._add_loop_constraints()

            if self.optimize:
                self._add_objective()

            result = self.solver.check()

            if result == sat:
                return self._extract_solution()
            elif result == unsat:
                return Schedule(
                    status=SolveStatus.UNSATISFIABLE,
                    component_name=self.model.component_name,
                    message="No feasible schedule exists",
                )
            else:
                return Schedule(
                    status=SolveStatus.UNKNOWN,
                    component_name=self.model.component_name,
                    message="Solver returned unknown",
                )

        except Exception as e:
            return Schedule(
                status=SolveStatus.ERROR,
                component_name=self.model.component_name,
                message=str(e),
            )

    def _create_variables(self):
        """Create Z3 integer variables."""
        for var in self.model.timing_vars:
            self.z3_vars[var] = Int(var)
        for var in self.model.binding_vars:
            self.z3_vars[var] = Int(var)
        for var in self.model.sched_vars:
            self.z3_vars[var] = Int(var)

    def _add_difference_constraints(self):
        """Add pure difference constraints (skip loop-specific ones handled elsewhere)."""
        _skip_kinds = {
            ConstraintType.RESOURCE,
            ConstraintType.BINDING_EQUALITY,
            ConstraintType.LOOP_RECURRENCE,   # handled in _add_loop_constraints
            ConstraintType.LOOP_MRT,          # replaced by modulo non-overlap
            ConstraintType.LOOP_II_NON_NEG,   # II is a constant, not a variable
            ConstraintType.LOOP_END_SINK,     # handled in _add_loop_constraints
            ConstraintType.LOOP_END_DEF,      # unused
        }
        for c in self.model.constraints:
            if c.kind in _skip_kinds:
                continue
            if not c.lhs:
                continue

            lhs = self.z3_vars.get(c.lhs)
            if lhs is None:
                continue

            if c.rhs:
                rhs = self.z3_vars.get(c.rhs)
                if rhs is None:
                    continue
                rhs_expr = rhs + c.constant
            else:
                rhs_expr = IntVal(c.constant)

            if c.is_equality:
                self.solver.add(lhs == rhs_expr)
            elif c.is_lower_bound:
                self.solver.add(lhs >= rhs_expr)
            else:
                self.solver.add(lhs <= rhs_expr)

    def _add_resource_constraints(self):
        """
        Add resource constraints using binary assignment variables.

        For pools with known size (max_instances = N):
          Binary x[op][unit] vars enforce assignment; non-overlap via Implies.

        For pools with unknown size (max_instances = None, i.e. * ?):
          Use integer op.i binding vars directly. Non-overlap via
          Implies(op1.i == op2.i, t1+lat <= t2 OR t2+lat <= t1).
          Pool size is minimized as a low-priority objective.
        """
        for c in self.model.by_type(ConstraintType.RESOURCE):
            pool_name = c.resource
            ops = c.operations
            max_units = c.max_concurrent
            latency = c.latency

            if max_units is None:
                # Unknown size: enforce non-overlap via integer binding vars
                for i, op1 in enumerate(ops):
                    t1 = self.z3_vars.get(f"{op1}.t")
                    b1 = self.z3_vars.get(f"{op1}.i")
                    if t1 is None or b1 is None:
                        continue
                    self.solver.add(b1 >= 0)
                    for op2 in ops[i+1:]:
                        t2 = self.z3_vars.get(f"{op2}.t")
                        b2 = self.z3_vars.get(f"{op2}.i")
                        if t2 is None or b2 is None:
                            continue
                        non_overlap = Or(t1 + latency <= t2, t2 + latency <= t1)
                        self.solver.add(Implies(b1 == b2, non_overlap))
                continue

            # Create binary variables: x[op][unit] = 1 if op uses unit
            self.z3_resource_vars[pool_name] = {}
            for op in ops:
                self.z3_resource_vars[pool_name][op] = {}
                for unit in range(max_units):
                    var_name = f"x_{op}_{pool_name}_{unit}"
                    self.z3_resource_vars[pool_name][op][unit] = Int(var_name)
                    x = self.z3_resource_vars[pool_name][op][unit]
                    self.solver.add(x >= 0)
                    self.solver.add(x <= 1)

                unit_sum = Sum([self.z3_resource_vars[pool_name][op][u] for u in range(max_units)])
                self.solver.add(unit_sum == 1)

            # Non-overlapping for same unit
            for i, op1 in enumerate(ops):
                t1 = self.z3_vars.get(f"{op1}.t")
                if t1 is None:
                    continue
                for j, op2 in enumerate(ops):
                    if j <= i:
                        continue
                    t2 = self.z3_vars.get(f"{op2}.t")
                    if t2 is None:
                        continue
                    for unit in range(max_units):
                        x1 = self.z3_resource_vars[pool_name][op1][unit]
                        x2 = self.z3_resource_vars[pool_name][op2][unit]
                        both_use_unit = And(x1 == 1, x2 == 1)
                        non_overlap = Or(t1 + latency <= t2, t2 + latency <= t1)
                        self.solver.add(Implies(both_use_unit, non_overlap))

    def _add_binding_constraints(self):
        """Add binding constraints (BINDING_EQUALITY and BINDING_EXPLICIT)."""
        for c in self.model.by_type(ConstraintType.BINDING_EQUALITY):
            ops = c.operations
            if len(ops) < 2:
                continue
            pool_name = None
            for op in ops:
                op_info = self.model.operations.get(op)
                if op_info and op_info.resource in self.z3_resource_vars:
                    pool_name = op_info.resource
                    break
            if pool_name is None:
                continue
            first_op = ops[0]
            for op in ops[1:]:
                if first_op not in self.z3_resource_vars.get(pool_name, {}):
                    continue
                if op not in self.z3_resource_vars.get(pool_name, {}):
                    continue
                for unit in self.z3_resource_vars[pool_name][first_op]:
                    x1 = self.z3_resource_vars[pool_name][first_op][unit]
                    x2 = self.z3_resource_vars[pool_name][op][unit]
                    self.solver.add(x1 == x2)

        for c in self.model.by_type(ConstraintType.BINDING_EXPLICIT):
            if not c.lhs or not c.lhs.endswith('.i'):
                continue
            op_name = c.lhs[:-2]
            explicit_unit = c.constant
            op_info = self.model.operations.get(op_name)
            if op_info is None:
                continue
            pool_name = op_info.resource
            if pool_name not in self.z3_resource_vars:
                continue
            if op_name not in self.z3_resource_vars[pool_name]:
                continue
            unit_vars = self.z3_resource_vars[pool_name][op_name]
            if explicit_unit in unit_vars:
                self.solver.add(unit_vars[explicit_unit] == 1)

    def _add_non_overlap_constraints(self):
        """Add pairwise non-overlap constraints for flat (non-loop) ops."""
        for c in self.model.by_type(ConstraintType.NON_OVERLAP):
            ops = c.operations
            if len(ops) != 2:
                continue
            op1, op2 = ops
            latency = c.latency
            pool_name = c.resource

            t1 = self.z3_vars.get(f"{op1}.t")
            t2 = self.z3_vars.get(f"{op2}.t")
            if t1 is None or t2 is None:
                continue

            has_resource_vars = (
                pool_name in self.z3_resource_vars and
                op1 in self.z3_resource_vars.get(pool_name, {}) and
                op2 in self.z3_resource_vars.get(pool_name, {})
            )

            if has_resource_vars:
                for unit in self.z3_resource_vars[pool_name][op1]:
                    x1 = self.z3_resource_vars[pool_name][op1][unit]
                    x2 = self.z3_resource_vars[pool_name][op2][unit]
                    both_use_unit = And(x1 == 1, x2 == 1)
                    non_overlap = Or(t1 + latency <= t2, t2 + latency <= t1)
                    self.solver.add(Implies(both_use_unit, non_overlap))
            else:
                non_overlap = Or(t1 + latency <= t2, t2 + latency <= t1)
                self.solver.add(non_overlap)

    def _add_loop_constraints(self):
        """
        Add loop-specific constraints with II substituted as a constant.

        For each loop region:
        1. Inter-iteration recurrence edges (LOOP_RECURRENCE with ii_dist):
              prod.t - cons.t <= II*dist - lat
           With II fixed: prod.t <= cons.t + II*dist - lat

        2. LOOP_END_SINK: end_var >= body.t + body_lat + II*(trip_count-1)
           With II fixed: pure difference constraint.

        3. Modulo non-overlap (from LoopModuloInfo.body_op_pairs):
           For each pair (a, b) on same pool unit:
              gap = (tb - ta) % II
              gap >= lat  AND  II - gap >= lat
           Conditioned on same unit assignment (via binary resource vars or
           integer binding vars).
        """
        # 1. Inter-iteration recurrence edges
        for c in self.model.by_type(ConstraintType.LOOP_RECURRENCE):
            if c.ii_dist is None:
                # Old-style lower bound on II var — skip (II is now a constant)
                continue
            # Find which loop region this belongs to via the ops
            # Use the first modulo info whose region contains the producer op
            ii_val = self._find_ii_for_constraint(c)
            if ii_val is None:
                continue
            lhs = self.z3_vars.get(c.lhs)
            rhs = self.z3_vars.get(c.rhs)
            if lhs is None or rhs is None:
                continue
            # prod.t <= cons.t + II*dist - lat
            bound = rhs + ii_val * c.ii_dist + c.constant  # c.constant = -lat
            self.solver.add(lhs <= bound)

        # 2. LOOP_END_SINK with fixed II
        # Formula: end >= Loop.N.t + (body.t - Loop.N.t) + lat + II*(trip_count-1)
        #        = end >= body.t + lat + II*(trip_count-1)  (loop_t cancels)
        for c in self.model.by_type(ConstraintType.LOOP_END_SINK):
            lhs = self.z3_vars.get(c.lhs)
            body_z3 = self.z3_vars.get(c.body_tvar)
            loop_start_z3 = self.z3_vars.get(c.rhs) if c.rhs else IntVal(0)
            if lhs is None or body_z3 is None:
                continue
            if c.ii_var is not None:
                # Find the fixed II for this loop
                ii_val = self._find_ii_by_ii_var(c.ii_var)
                if ii_val is None:
                    continue
            else:
                ii_val = c.constant  # fixed II from region.ii
            # loop_start_z3 appears in both sides and cancels; kept for clarity
            self.solver.add(lhs >= loop_start_z3 + (body_z3 - loop_start_z3) + c.body_latency + ii_val * (c.trip_count - 1))

        # 3. Modulo non-overlap
        for info in self.model.loop_modulo_infos:
            ii_val = self.loop_iis.get(info.region_idx)
            if ii_val is None:
                # Fixed II from region spec
                region = self.model.loop_regions[info.region_idx]
                ii_val = region.ii
            if ii_val is None:
                continue

            for pool_name, pairs in info.body_op_pairs.items():
                for op1, op2, lat in pairs:
                    t1 = self.z3_vars.get(f"{op1}.t")
                    t2 = self.z3_vars.get(f"{op2}.t")
                    b1 = self.z3_vars.get(f"{op1}.i")
                    b2 = self.z3_vars.get(f"{op2}.i")
                    if t1 is None or t2 is None:
                        continue

                    # Modulo non-overlap via Z3 native %
                    # gap = (t2 - t1) % II  — forward circular distance in [0, II)
                    # Both directions must have sufficient separation:
                    #   gap >= lat  (b at least lat after a)
                    #   II - gap >= lat  (a at least lat after b, wrap-around)
                    gap = (t2 - t1) % ii_val
                    modulo_ok = And(gap >= lat, ii_val - gap >= lat)

                    # Conditioned on same unit
                    if b1 is not None and b2 is not None:
                        # Unknown pool size: same unit = b1 == b2
                        self.solver.add(Implies(b1 == b2, modulo_ok))
                    elif (pool_name in self.z3_resource_vars and
                          op1 in self.z3_resource_vars[pool_name] and
                          op2 in self.z3_resource_vars[pool_name]):
                        # Known pool size: conditioned per unit
                        for unit in self.z3_resource_vars[pool_name][op1]:
                            x1 = self.z3_resource_vars[pool_name][op1][unit]
                            x2 = self.z3_resource_vars[pool_name][op2][unit]
                            both = And(x1 == 1, x2 == 1)
                            self.solver.add(Implies(both, modulo_ok))
                    else:
                        # No binding info: unconditional
                        self.solver.add(modulo_ok)

    def _find_ii_for_constraint(self, c: Constraint) -> Optional[int]:
        """Find the fixed II value for a LOOP_RECURRENCE constraint."""
        # Match by op name prefix to a loop region
        op_name = c.lhs[:-2] if c.lhs.endswith('.t') else c.lhs  # strip .t
        for info in self.model.loop_modulo_infos:
            region = self.model.loop_regions[info.region_idx]
            if op_name in region.body_ops:
                ii_val = self.loop_iis.get(info.region_idx)
                if ii_val is None:
                    ii_val = region.ii
                return ii_val
        return None

    def _find_ii_by_ii_var(self, ii_var: str) -> Optional[int]:
        """Find the fixed II value given an ii_var name like 'Loop.0.II'."""
        for info in self.model.loop_modulo_infos:
            if info.ii_var == ii_var:
                ii_val = self.loop_iis.get(info.region_idx)
                if ii_val is None:
                    region = self.model.loop_regions[info.region_idx]
                    ii_val = region.ii
                return ii_val
        return None

    def _add_objective(self):
        """Minimize makespan (max op end time), then loop-end vars."""
        if not self.z3_vars:
            return

        makespan = Int("makespan")
        for var_name, var in self.z3_vars.items():
            if not var_name.endswith('.t'):
                continue
            op_name = var_name[:-2]
            op_info = self.model.operations.get(op_name)
            if op_info is None:
                continue
            latency = self.model.latencies.get(op_info.resource, 1)
            self.solver.add(makespan >= var + latency)
        self.solver.minimize(makespan)

        # Secondary: minimize loop-end vars so they're tight
        for var_name, var in self.z3_vars.items():
            if var_name.startswith("Loop.") and var_name.endswith(".end"):
                self.solver.minimize(var)

        # Tertiary: minimize pool sizes for unknown-size pools
        for c in self.model.by_type(ConstraintType.RESOURCE):
            if c.max_concurrent is not None:
                continue
            binding_z3_vars = [self.z3_vars.get(f"{op}.i") for op in c.operations]
            binding_z3_vars = [v for v in binding_z3_vars if v is not None]
            if not binding_z3_vars:
                continue
            pool_max = Int(f"pool_size_{c.resource}")
            for bv in binding_z3_vars:
                self.solver.add(pool_max >= bv + 1)
            self.solver.minimize(pool_max)

    def _extract_solution(self) -> Schedule:
        """Extract solution from Z3 model."""
        z3_model = self.solver.model()

        assignments = {}
        for var_name, z3_var in self.z3_vars.items():
            val = z3_model.eval(z3_var, model_completion=True)
            assignments[var_name] = val.as_long() if hasattr(val, 'as_long') else int(str(val))

        # Inject fixed II values into assignments for display
        for info in self.model.loop_modulo_infos:
            ii_val = self.loop_iis.get(info.region_idx)
            if ii_val is None:
                region = self.model.loop_regions[info.region_idx]
                ii_val = region.ii
            if ii_val is not None:
                assignments[info.ii_var] = ii_val

        # Extract resource assignments from binary variables
        resource_assignments = {}
        for pool_name, pool_vars in self.z3_resource_vars.items():
            for op, unit_vars in pool_vars.items():
                for unit, x_var in unit_vars.items():
                    val = z3_model.eval(x_var, model_completion=True)
                    if hasattr(val, 'as_long') and val.as_long() == 1:
                        resource_assignments[op] = unit
                    elif str(val) == '1':
                        resource_assignments[op] = unit

        # Update .i binding variables to reflect actual unit assignments
        for op, unit in resource_assignments.items():
            binding_var = f"{op}.i"
            if binding_var in assignments:
                assignments[binding_var] = unit

        # Compute makespan
        makespan = 0
        for var_name, start in assignments.items():
            if not var_name.endswith('.t'):
                continue
            op_name = var_name[:-2]
            op_info = self.model.operations.get(op_name)
            if op_info is None:
                continue
            latency = self.model.latencies.get(op_info.resource, 1)
            makespan = max(makespan, start + latency)

        pool_latencies = {}
        for res_name, res_info in self.model.resources.items():
            pool_latencies[res_name] = res_info.latency

        return Schedule(
            status=SolveStatus.OPTIMAL if self.optimize else SolveStatus.SATISFIABLE,
            component_name=self.model.component_name,
            assignments=assignments,
            makespan=makespan,
            start_time=min(assignments.values()) if assignments else 0,
            resource_assignments=resource_assignments,
            pool_latencies=pool_latencies,
            tvar_aliases=dict(self.model.tvar_aliases),
        )


def solve_sdc(model: SDCModel, optimize: bool = True) -> Schedule:
    """
    Solve SDC model and return schedule.

    If the model has loop regions with solver-determined II (region.ii is None),
    iterates II from MII upward until a satisfiable schedule is found.
    """
    if not Z3_AVAILABLE:
        return Schedule(
            status=SolveStatus.ERROR,
            component_name=model.component_name,
            message="z3-solver not installed. Run: pip install z3-solver",
        )

    # Collect loop regions that need II iteration
    variable_ii_infos = [
        info for info in model.loop_modulo_infos
        if model.loop_regions[info.region_idx].ii is None
    ]

    if not variable_ii_infos:
        # No variable-II loops: single solve
        return Z3Solver(model, loop_iis={}, optimize=optimize).solve()

    # Compute upper bound: sum of all body op latencies across all loops
    # (trivially schedulable sequentially)
    upper_bound = 0
    for info in variable_ii_infos:
        region = model.loop_regions[info.region_idx]
        for op in region.body_ops.values():
            upper_bound += model.latencies.get(op.resource, 1)
    upper_bound = max(upper_bound, 1)

    # Start all variable-II loops at their MII
    current_iis = {info.region_idx: info.mii for info in variable_ii_infos}

    last_schedule = None
    while True:
        # Check termination: all IIs within bounds
        if any(current_iis[info.region_idx] > upper_bound for info in variable_ii_infos):
            return Schedule(
                status=SolveStatus.UNSATISFIABLE,
                component_name=model.component_name,
                message=f"No schedule found up to II={upper_bound}",
            )

        schedule = Z3Solver(model, loop_iis=dict(current_iis), optimize=optimize).solve()

        if schedule.status in (SolveStatus.OPTIMAL, SolveStatus.SATISFIABLE):
            return schedule

        if schedule.status == SolveStatus.UNSATISFIABLE:
            # Increment all variable-II loops by 1 and retry
            # (simple strategy: increment all together)
            for info in variable_ii_infos:
                current_iis[info.region_idx] += 1
        else:
            # UNKNOWN or ERROR: give up
            return schedule
