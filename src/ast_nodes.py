"""
AST Node Definitions for Jalapeno

This module defines all AST node types used by the Jalapeno compiler.
Nodes are organized by category for clarity.
"""

from dataclasses import dataclass
from typing import List, Optional, Any


# ============================================================================
# Expression Nodes
# ============================================================================

@dataclass
class Number:
    """Numeric literal"""
    value: int


@dataclass
class Var:
    """Variable reference"""
    name: str


@dataclass
class BinOp:
    """Binary operation: left op right"""
    op: str
    left: Any
    right: Any


# ============================================================================
# Time System Nodes
# ============================================================================

@dataclass
class Event:
    """Event identifier: 'G, 'clk, etc."""
    name: str


@dataclass
class SchedVar:
    """
    Scheduler variable for HLS partial scheduling.
    - Named: ?t_b1, ?foo (name is set)
    - Anonymous: ? (name is None)
    """
    name: Optional[str] = None


@dataclass
class RangeOffset:
    """
    Range offset for flexible timing constraints.
    - [lo..hi]: bounded range
    - [lo..]: lower-bounded (hi is None)
    - [..hi]: upper-bounded (lo is None)
    """
    lo: Optional[Any] = None
    hi: Optional[Any] = None


@dataclass
class Time:
    """
    Time expression - represents a point in time.

    Can be based on:
    - Event: 'G, 'G+1, 'G+[4..]
    - SchedVar: ?t, ?t+1, ?t+[2..4], ?
    - Named bind: ?name=time  (alias names the time point, binds it to another time)

    Fields:
    - event: Event base (mutually exclusive with sched_var)
    - sched_var: SchedVar base (mutually exclusive with event)
    - offset: Optional offset (Number, BinOp, or RangeOffset)
    - alias: Optional name from ?name=time syntax
    - loop_end: True for ?alias.e syntax (loop-end time = start + II * N)
    """
    event: Optional[Event] = None
    sched_var: Optional[SchedVar] = None
    offset: Optional[Any] = None
    alias: Optional[str] = None
    loop_end: bool = False


@dataclass
class Interval:
    """Time interval: [start, end]"""
    start: Time
    end: Time


@dataclass
class EventBind:
    """
    Event binding in abstract vars: 'G: 1 or 'G: ?
    - delay can be an expression (Number, Var, BinOp) or SchedVar
    """
    event: Event
    delay: Any


# ============================================================================
# Port and IO Nodes
# ============================================================================

@dataclass
class PortDef:
    """
    Port definition in component signature.
    - Signal port:  name: [interval] width
    - Bundle port:  name[N]: [interval] width
    - Interface port: name: interface[event]
    bundle_size is None for scalar ports, N for bundle-of-N ports.
    """
    name: str
    interval: Optional[Interval]
    width: Any
    is_interface: bool = False
    interface_event: Optional[Event] = None
    bundle_size: Optional[int] = None


@dataclass
class PortRef:
    """
    Port reference in expressions.
    - Qualified: instance.port
    - Simple: port (instance is None)
    """
    instance: Optional[str]
    port: str


@dataclass
class PortBundleAccess:
    """
    Bundle element access in port_expr position.
    - Simple:    bundle{i}       -> instance=None
    - Qualified: inst.bundle{i}  -> instance='inst'
    The index is an expression (Number, Var, BinOp).
    """
    instance: Optional[str]
    bundle: str
    index: Any


# ============================================================================
# Module Structure Nodes
# ============================================================================

@dataclass
class Param:
    """Parameter in component definition: [W, N]"""
    name: str


@dataclass
class Constraint:
    """Constraint in where clause: W > 0"""
    left: Any
    op: str
    right: Any


@dataclass
class Signature:
    """Component signature (shared by comp and extern)"""
    name: str
    params: List[Param]
    event_binds: List[EventBind]
    inputs: List[PortDef]
    outputs: List[PortDef]
    constraints: List[Constraint]


# ============================================================================
# Resource Pool Nodes
# ============================================================================

@dataclass
class PoolSize:
    """
    Pool size specification.
    - max_count: Maximum instances (None if unbounded)
    - is_unbounded: True if scheduler decides (?)
    """
    max_count: Optional[Any] = None  # expr for N, None for ?
    is_unbounded: bool = False


@dataclass
class Pool:
    """
    Pool declaration: pool A : Add[32] * N;
    - name: Pool identifier
    - module: Module type for instances
    - params: Module parameters [32]
    - size: PoolSize (max or unbounded)
    """
    name: str
    module: str
    params: List[Any]
    size: PoolSize


@dataclass
class BindingIndex:
    """
    Binding index for pool invocations.
    - kind: 'anon' (scheduler picks), 'named' (same name = same instance), 'explicit' (specific index)
    - value: None for anon, name string for named, expr for explicit
    """
    kind: str  # 'anon', 'named', 'explicit'
    value: Optional[Any] = None


# ============================================================================
# Command Nodes
# ============================================================================

@dataclass
class Instance:
    """Instance creation: A := new Add[32];"""
    name: str
    module: str
    params: List[Any]
    time_args: List[Time]
    args: List[Any]


@dataclass
class Invocation:
    """
    Instance invocation with optional binding.
    - Traditional: result := A<'G>(a, b);
    - With binding: result := A[?]<?t>(a, b);
    - binding: None for traditional, BindingIndex for pool access
    """
    name: str
    instance: str
    binding: Optional[BindingIndex]
    time_args: List[Time]
    args: List[Any]


@dataclass
class Connect:
    """Connection: out = result.out;"""
    dest: Any
    src: Any


@dataclass
class TimeLoop:
    """
    Pipelined time loop with modulo scheduling.

    Syntax: timeloop<?start> [II=4] for i in 0..N { ... }

    - var:   loop variable name ("i")
    - start: Optional Time — when iteration 0 fires:
               Time(sched_var=SchedVar('s')) for <?s>
               Time(event=Event('G'), offset=...) for <'G+2>
               None if omitted (scheduler decides freely)
    - ii:    Optional initiation interval:
               Number(n) if user-fixed [II=4]
               SchedVar(None) if solver-determined [II=?]
               None if [II=...] clause omitted (solver decides)
    - end:   trip count expression (exclusive upper bound)
    - body:  list of commands (invocations, pools, etc.)
    """
    var: str
    start: Optional[Any]   # Time | None
    ii: Optional[Any]      # Number | SchedVar(None) | None
    end: Any
    body: List[Any]


@dataclass
class SpaceLoop:
    """
    Structural space loop — pure duplication, no cross-iteration scheduling.

    Syntax: spaceloop<'G> for i in 0..N { ... }

    - var:   loop variable name
    - start: Time — when the loop body executes (required)
    - end:   upper bound expression (exclusive, lower bound always 0)
    - body:  list of commands
    """
    var: str
    start: Any   # Time
    end: Any
    body: List[Any]


@dataclass
class IfStmt:
    """If statement: if cond { ... } else { ... }"""
    condition: Any
    then_body: List[Any]
    else_body: Optional[List[Any]]


@dataclass
class BundleDecl:
    """
    Standalone bundle declaration inside a component body.

    Syntax: bundle twiddle[2]: ['G, 'G+1] 32;

    Declares a fixed-size array of wires sharing the same timing interval.
    bundle_size must be a statically-known integer literal.
    """
    name: str
    bundle_size: int
    interval: Any   # Interval
    width: Any


@dataclass
class ParamLet:
    """Let binding: let x = expr;"""
    name: str
    value: Any


# ============================================================================
# Top-Level Nodes
# ============================================================================

@dataclass
class Component:
    """Component definition: comp Name[...] { ... }"""
    signature: Signature
    commands: List[Any]


@dataclass
class ExternBlock:
    """Extern block: extern "path" { signatures }"""
    path: str
    signatures: List[Signature]


@dataclass
class Import:
    """Import statement: import "path";"""
    path: str


@dataclass
class File:
    """Top-level file containing imports and modules"""
    imports: List[Import]
    modules: List[Any]  # Component or ExternBlock
