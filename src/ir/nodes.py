"""
IR Node Definitions

Data classes representing the Scheduling IR — the compiler's internal
representation between the AST and the constraint solver.

Node hierarchy:
  SchedulingIR      top-level component IR
    PortTiming      timing of an input/output port (scalar or bundle)
    Resource        instance or pool resource
    Operation       a schedulable operation (from an invocation)
      TimingConstraint  timing annotation on an operation
    BundleInfo      a declared bundle wire array
    LoopRegion      a timeloop body (symbolic, not unrolled)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any


# ============================================================================
# Timing
# ============================================================================

@dataclass
class TimingConstraint:
    """
    Timing annotation on an operation, derived from time_args in an invocation.

    - Simple sched_var: ?t -> creates a variable to solve
    - Offset: ?t+1 -> relative constraint
    - Range: ?t+[2..4] or 'G+[4..] -> bounded constraint
    """
    sched_var: Optional[str] = None        # Named sched var (None if anonymous)

    base_event: Optional[str] = None       # Event base, e.g. "G" from 'G
    base_sched_var: Optional[str] = None

    offset: Optional[int] = None           # Fixed offset: ?t+1
    range_lo: Optional[int] = None         # Range lower bound: [lo..]
    range_hi: Optional[int] = None         # Range upper bound: [..hi]
    alias: Optional[str] = None            # Named alias from ?name=time syntax
    loop_end: bool = False                 # True for ?alias.e (loop-end time)

    def is_anonymous(self) -> bool:
        return self.sched_var is None and self.base_sched_var is None


@dataclass
class PortTiming:
    """
    Timing information for a component port.

    Times can be event-based ('G, 'G+1) or sched-var-based (?, ?+1).
    bundle_size is None for scalar ports, N for bundle-of-N ports.
    """
    name: str
    width: Optional[int] = None
    bundle_size: Optional[int] = None      # None = scalar, N = bundle of N wires

    start_event: Optional[str] = None
    start_sched_var: Optional[str] = None  # "" = anonymous ?
    start_offset: Optional[int] = None

    end_event: Optional[str] = None
    end_sched_var: Optional[str] = None
    end_offset: Optional[int] = None

    is_interface: bool = False

    def is_fixed(self) -> bool:
        """True if timing is fully event-based (not sched-var)."""
        return self.start_event is not None and self.start_sched_var is None


# ============================================================================
# Resources and Operations
# ============================================================================

@dataclass
class Resource:
    """A hardware resource — either a plain instance or a shared pool."""
    name: str
    module: str
    params: List[Any] = field(default_factory=list)
    latency: int = 1

    is_pool: bool = False
    max_instances: Optional[int] = None    # None = unbounded


@dataclass
class Operation:
    """
    A schedulable operation produced by an invocation.

    Named by resource + index: "A.0", "A.1", "A.2".
    The result wire name is the lhs of the invocation (e.g., "b1").
    """
    op_name: str                           # "A.0", "A.1"  (resource.index)
    resource: str                          # instance or pool name
    result: str                            # wire name produced (for dep tracking)
    timing: TimingConstraint
    inputs: List[str] = field(default_factory=list)

    binding: Optional[Any] = None          # BindingIndex for pool ops


# ============================================================================
# Bundle
# ============================================================================

@dataclass
class BundleInfo:
    """
    Metadata for a declared bundle wire array.

    Represents either:
    - A bundle port on the component signature (from PortDef.bundle_size)
    - A standalone bundle declaration in the body (from BundleDecl)

    All wires in the bundle share the same timing interval.
    """
    name: str
    bundle_size: int
    width: Optional[int] = None

    start_event: Optional[str] = None
    start_sched_var: Optional[str] = None
    start_offset: Optional[int] = None

    end_event: Optional[str] = None
    end_sched_var: Optional[str] = None
    end_offset: Optional[int] = None


# ============================================================================
# Loop Region
# ============================================================================

@dataclass
class LoopRegion:
    """
    IR node for a timeloop body — kept symbolic, not unrolled.

    Represents one canonical iteration. The modulo scheduler will use II
    and trip_count to reason about recurrence constraints across iterations.

    body_instances:    instance name -> latency  (from `new` decls)
    body_pools:        pool name -> Resource      (from `pool` decls)
    body_bundles:      bundle name -> BundleInfo  (from `bundle` decls)
    body_ops:          op_name -> Operation       (invocations in one iteration)
    body_result_to_op: result wire -> op_name
    sched_vars:        scheduler vars introduced in the body
    """
    loop_var: str
    start: TimingConstraint
    ii: Optional[int]
    trip_count: int

    body_instances: Dict[str, int] = field(default_factory=dict)
    body_pools: Dict[str, Resource] = field(default_factory=dict)
    body_bundles: Dict[str, BundleInfo] = field(default_factory=dict)
    body_ops: Dict[str, Operation] = field(default_factory=dict)
    body_result_to_op: Dict[str, str] = field(default_factory=dict)
    sched_vars: Set[str] = field(default_factory=set)
    # Intra-body connect statements: list of (dest_expr, src_expr) as strings
    # e.g. ("s{i+1}", "r.out") for bundle wire chain assignments
    body_connects: List[tuple] = field(default_factory=list)
    # alias -> loop-end var name, e.g. "tloop" -> "?tloop_end"
    # Populated when a post-loop op references ?alias.e
    loop_end_vars: Dict[str, str] = field(default_factory=dict)

    _op_counters: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# Top-level IR
# ============================================================================

@dataclass
class SchedulingIR:
    """
    Top-level IR for a component — everything needed for scheduling.

    Flat operations (from top-level invocations and unrolled spaceloops)
    live in `operations`.  Timeloop bodies live in `loops` as LoopRegions.
    Bundle declarations (body-level) live in `bundles`.
    """
    name: str

    start_event: str = "G"
    ii: Optional[int] = None

    inputs: Dict[str, PortTiming] = field(default_factory=dict)
    outputs: Dict[str, PortTiming] = field(default_factory=dict)

    # output_port -> (producer_op_name, port_name)  [op result connections]
    output_connections: Dict[str, tuple] = field(default_factory=dict)
    # output_bundle_port -> source_bundle_name  [whole-bundle wire connections: sum = s]
    bundle_connections: Dict[str, str] = field(default_factory=dict)

    pools: Dict[str, Resource] = field(default_factory=dict)
    operations: Dict[str, Operation] = field(default_factory=dict)
    result_to_op: Dict[str, str] = field(default_factory=dict)
    module_latencies: Dict[str, int] = field(default_factory=dict)
    sched_vars: Set[str] = field(default_factory=set)
    bundles: Dict[str, BundleInfo] = field(default_factory=dict)
    # Compile-time constants: wire name -> integer value (from `new Const[W, V]`)
    # These carry no timing; they are folded into op inputs as "#V" literals.
    const_values: Dict[str, int] = field(default_factory=dict)
    # Bundle element init assignments outside loop bodies: "s{0}" -> "#0" or "wire"
    bundle_inits: Dict[str, str] = field(default_factory=dict)

    _op_counters: Dict[str, int] = field(default_factory=dict)

    loops: List[LoopRegion] = field(default_factory=list)
