"""
SDC Model — constraint data classes

Defines:
  ConstraintType   enum of all constraint categories
  Constraint       a single scheduling constraint
  SDCModel         the complete constraint model for one component
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

from ir.nodes import Resource, Operation, LoopRegion


class ConstraintType(Enum):
    """Categories of scheduling constraints."""
    INPUT_WINDOW       = "input_window"       # op.t in [input_start, input_end)
    OP_TO_OP           = "op_to_op"           # consumer.t >= producer.t + latency
    OUTPUT_DEADLINE    = "output_deadline"    # producer.t + latency <= deadline

    TIMING_EQUALITY    = "timing_equality"    # op.t == base + offset
    TIMING_LOWER       = "timing_lower"       # op.t >= base + lo
    TIMING_UPPER       = "timing_upper"       # op.t <= base + hi

    RESOURCE           = "resource"           # at most N ops active at once
    BINDING_EQUALITY   = "binding_equality"   # ops share same instance
    BINDING_EXPLICIT   = "binding_explicit"   # op.i == specific_index

    NON_NEGATIVE       = "non_negative"       # op.t >= 0
    SCHED_VAR_LINK     = "sched_var_link"     # link ops with same ?var
    NON_OVERLAP        = "non_overlap"        # ops on same unit can't overlap

    # Loop region constraints (modulo scheduling)
    LOOP_INTRA_DEP     = "loop_intra_dep"     # consumer.t >= producer.t + lat (within one iter)
    LOOP_RECURRENCE    = "loop_recurrence"    # II >= ceil(latency / distance)  (loop-carried edge)
    LOOP_MRT           = "loop_mrt"           # II >= ceil(k*lat/N)  (modulo reservation table)
    LOOP_II_NON_NEG    = "loop_ii_non_neg"    # II >= 1
    LOOP_END_DEF       = "loop_end_def"       # (unused — kept for skip-set compatibility)
    LOOP_END_SINK      = "loop_end_sink"      # ?alias_end >= sink_op.t + lat + II*(N-1)


@dataclass
class Constraint:
    """
    A single scheduling constraint, normalised to:
      lhs >= rhs + constant   (lower bound)
      lhs <= rhs + constant   (upper bound)
      lhs == rhs + constant   (equality)
    """
    kind: ConstraintType

    lhs: str = ""
    rhs: Optional[str] = None
    constant: int = 0

    is_lower_bound: bool = True
    is_equality: bool = False

    # For resource constraints
    resource: Optional[str] = None
    operations: List[str] = field(default_factory=list)
    max_concurrent: Optional[int] = None
    latency: int = 1

    binding_var: Optional[str] = None
    reason: str = ""

    # For inter-iteration recurrence: su - sv <= II*ii_dist - lat
    # ii_dist is the loop-carried distance (substituted at solve time with fixed II)
    ii_dist: Optional[int] = None

    # For LOOP_END_DEF (unused) and LOOP_END_SINK
    trip_count: Optional[int] = None
    ii_var: Optional[str] = None
    # For LOOP_END_SINK: ?end_var >= body_tvar + body_latency + ii_var*(trip_count-1)
    body_tvar: Optional[str] = None
    body_latency: int = 0

    def __str__(self) -> str:
        if self.kind == ConstraintType.RESOURCE:
            return (f"resource({self.resource}): at most {self.max_concurrent} "
                    f"of {self.operations} active")
        if self.kind == ConstraintType.BINDING_EQUALITY:
            return f"binding({self.binding_var}): {self.operations} share same unit"
        if self.kind == ConstraintType.NON_OVERLAP:
            return (f"non_overlap({self.operations[0]}, {self.operations[1]}): "
                    f"latency({self.resource})={self.latency}")
        if self.kind == ConstraintType.LOOP_END_DEF:
            if self.ii_var is not None:
                rhs = f"{self.constant} + {self.ii_var} * {self.trip_count}"
            else:
                rhs = str(self.constant)
            return f"{self.lhs} == {rhs}"

        if self.rhs:
            rhs_str = self.rhs
            if self.constant > 0:
                rhs_str += f" + {self.constant}"
            elif self.constant < 0:
                rhs_str += f" - {-self.constant}"
        else:
            rhs_str = str(self.constant)

        op = "==" if self.is_equality else (">=" if self.is_lower_bound else "<=")
        return f"{self.lhs} {op} {rhs_str}"


@dataclass
class LoopModuloInfo:
    """
    Per-loop info needed for the II iteration loop in the solver.

    mii: minimum II lower bound (from recurrence + MRT, analytical)
    body_op_pairs: per pool_name -> list of (op1, op2, latency) pairs
                   that need modulo non-overlap constraints
    region_idx: index into SDCModel.loop_regions
    ii_var: the Loop.N.II variable name (kept for display only)
    """
    region_idx: int
    mii: int
    ii_var: str
    # pool_name -> [(op1_name, op2_name, latency)]
    body_op_pairs: Dict[str, List[Tuple[str, str, int]]] = field(default_factory=dict)
    # pool_name -> latency: pools that have any body ops (self-overlap: II >= lat)
    body_self_overlap: Dict[str, int] = field(default_factory=dict)


@dataclass
class SDCModel:
    """
    Complete constraint model for scheduling a single component.

    Variables:
      {op}.t  — timing variable for each operation
      {op}.i  — binding variable for pool operations
      ?{name} — user-defined scheduler timing variable
    """
    component_name: str

    timing_vars: Set[str] = field(default_factory=set)
    binding_vars: Set[str] = field(default_factory=set)
    sched_vars: Set[str] = field(default_factory=set)

    constraints: List[Constraint] = field(default_factory=list)

    resources: Dict[str, Resource] = field(default_factory=dict)
    operations: Dict[str, Operation] = field(default_factory=dict)
    latencies: Dict[str, int] = field(default_factory=dict)

    sched_var_to_ops: Dict[str, List[str]] = field(default_factory=dict)

    # Loop regions carried through for display and solving
    loop_regions: List[LoopRegion] = field(default_factory=list)
    # region index -> II variable name (display only, e.g. "Loop.0.II")
    loop_ii_vars: Dict[int, str] = field(default_factory=dict)
    # Per-loop modulo scheduling info (MII, body op pairs for non-overlap)
    loop_modulo_infos: List[LoopModuloInfo] = field(default_factory=list)

    # op_timing_var -> alias_name, e.g. "A.0.t" -> "ta"
    # Populated by SDCGenerator for display annotation.
    tvar_aliases: Dict[str, str] = field(default_factory=dict)

    def add(self, constraint: Constraint):
        self.constraints.append(constraint)

    def by_type(self, kind: ConstraintType) -> List[Constraint]:
        return [c for c in self.constraints if c.kind == kind]

    def print_summary(self):
        """Print constraint summary."""
        print(f"\nSDC Model: {self.component_name}")
        print(f"  Timing variables: {sorted(self.timing_vars)}")
        if self.binding_vars:
            print(f"  Binding variables: {sorted(self.binding_vars)}")
        if self.sched_vars:
            print(f"  Scheduler variables: {sorted(self.sched_vars)}")
        print(f"  Total constraints: {len(self.constraints)}")

        self._print_data_deps()
        self._print_timing_constraints()
        self._print_resource_constraints()
        self._print_binding_constraints()
        self._print_loop_constraints()
        self._print_non_negative()

    # ------------------------------------------------------------------

    def _print_data_deps(self):
        input_cs  = self.by_type(ConstraintType.INPUT_WINDOW)
        op_cs     = self.by_type(ConstraintType.OP_TO_OP)
        output_cs = self.by_type(ConstraintType.OUTPUT_DEADLINE)
        if not (input_cs or op_cs or output_cs):
            return

        print("\n  Data Dependencies:")
        if input_cs:
            print("    (a) Input availability:")
            self._print_input_constraints(input_cs)
        if op_cs:
            print("    (b) Op-to-op:")
            for c in op_cs:
                print(f"        {c}  // {c.reason}")
        if output_cs:
            print("    (c) Output deadlines:")
            for c in output_cs:
                print(f"        {c}  // {c.reason}")

    def _print_timing_constraints(self):
        eq     = self.by_type(ConstraintType.TIMING_EQUALITY)
        lo     = self.by_type(ConstraintType.TIMING_LOWER)
        hi     = self.by_type(ConstraintType.TIMING_UPPER)
        links  = self.by_type(ConstraintType.SCHED_VAR_LINK)
        if not (eq or lo or hi or links):
            return

        print("\n  User Timing Constraints:")
        for c in eq + lo + hi:
            print(f"    {c}  // {c.reason}")
        if links:
            # Group by rhs (the resolved variable or aliased op.t) for readability
            by_rhs: Dict[str, List[Constraint]] = {}
            for c in links:
                rhs = c.rhs or "constant"
                by_rhs.setdefault(rhs, []).append(c)
            for rhs in sorted(by_rhs):
                print(f"    {rhs}:")
                for c in by_rhs[rhs]:
                    print(f"      {c}  // {c.reason}")

    def _print_resource_constraints(self):
        rcs = self.by_type(ConstraintType.RESOURCE)
        non_overlap = self.by_type(ConstraintType.NON_OVERLAP)
        if rcs:
            print("\n  Resource Constraints:")
            for c in rcs:
                bvars = [f"{op}.i" for op in c.operations]
                print(f"    0 <= {', '.join(bvars)} < {c.max_concurrent}"
                      f"  // pool {c.resource} has {c.max_concurrent} units")
        if non_overlap:
            print("\n  Non-Overlap Constraints:")
            for c in non_overlap:
                op1, op2 = c.operations
                print(f"    {op1}.i == {op2}.i => "
                      f"({op1}.t + latency({c.resource}) <= {op2}.t OR "
                      f"{op2}.t + latency({c.resource}) <= {op1}.t)")

    def _print_binding_constraints(self):
        eq  = self.by_type(ConstraintType.BINDING_EQUALITY)
        exp = self.by_type(ConstraintType.BINDING_EXPLICIT)
        if eq or exp:
            print("\n  Binding Constraints:")
            for c in eq + exp:
                print(f"    {c}  // {c.reason}")

    def _print_loop_constraints(self):
        intra  = self.by_type(ConstraintType.LOOP_INTRA_DEP)
        recur  = self.by_type(ConstraintType.LOOP_RECURRENCE)
        sink   = self.by_type(ConstraintType.LOOP_END_SINK)
        if not (intra or recur or sink or self.loop_modulo_infos):
            return

        print("\n  Loop Region Constraints:")
        for idx, region in enumerate(self.loop_regions):
            if region.ii is not None:
                ii_desc = f"II={region.ii} (fixed)"
            else:
                mii = self.loop_modulo_infos[idx].mii if idx < len(self.loop_modulo_infos) else "?"
                ii_desc = f"II=? (solver iterates from MII={mii})"
            print(f"\n    [{idx}] loop var={region.loop_var}, "
                  f"trip_count={region.trip_count}, {ii_desc}")

        if intra:
            print("\n    (a) Intra-iteration dependencies:")
            for c in intra:
                print(f"        {c}  // {c.reason}")

        if recur:
            print("\n    (b) Recurrence (inter-iteration, loop-carried):")
            for c in recur:
                if c.ii_dist is not None:
                    lat = -c.constant
                    print(f"        {c.lhs} - {c.rhs} <= II*{c.ii_dist} - {lat}  // {c.reason}")
                else:
                    print(f"        {c.lhs} >= {c.constant}  // {c.reason}")

        if sink:
            print("\n    (c) Loop-end lower bounds (per sink op):")
            for c in sink:
                ii_str = c.ii_var if c.ii_var else str(c.constant)
                n = (c.trip_count - 1) if c.trip_count is not None else "N-1"
                loop_t = c.rhs or "Loop.?.t"
                print(f"        {c.lhs} >= {loop_t} + ({c.body_tvar} - {loop_t})"
                      f" + {c.body_latency} + {ii_str}*{n}  // {c.reason}")

        # Modulo non-overlap (from LoopModuloInfo — added by solver, shown here for completeness)
        for info in self.loop_modulo_infos:
            if not info.body_op_pairs:
                continue
            print(f"\n    (d) Modulo non-overlap (II substituted per iteration):")
            for pool_name, pairs in info.body_op_pairs.items():
                for op1, op2, lat in pairs:
                    print(f"        [{pool_name}] gap = ({op2}.t - {op1}.t) % II;"
                          f"  gap >= {lat}  AND  II - gap >= {lat}")

    def _print_non_negative(self):
        nn = self.by_type(ConstraintType.NON_NEGATIVE)
        if nn:
            print(f"\n  Non-negative: {', '.join(c.lhs for c in nn)} >= 0")

    def _print_input_constraints(self, constraints: List[Constraint]):
        by_op: Dict[str, List[Constraint]] = {}
        for c in constraints:
            by_op.setdefault(c.lhs, []).append(c)

        for op, cs in sorted(by_op.items()):
            by_input: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
            for c in cs:
                if "[input:" in c.reason:
                    input_name = c.reason.split("[input:")[1].split(",")[0].strip()
                else:
                    input_name = "unknown"
                lo, hi = by_input.get(input_name, (None, None))
                if c.is_lower_bound:
                    lo = c.constant
                else:
                    hi = c.constant
                by_input[input_name] = (lo, hi)

            for input_name, (lo, hi) in by_input.items():
                if lo is not None and hi is not None:
                    if lo == hi:
                        print(f"        {op} == {lo}  // from {input_name}: [{lo}, {hi+1})")
                    else:
                        print(f"        {lo} <= {op} <= {hi}  // from {input_name}: [{lo}, {hi+1})")
                elif lo is not None:
                    print(f"        {op} >= {lo}  // from {input_name}: [{lo}, ...)")
                elif hi is not None:
                    print(f"        {op} <= {hi}  // from {input_name}: [..., {hi+1})")
