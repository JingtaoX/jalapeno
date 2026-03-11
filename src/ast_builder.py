"""
AST Builder

Transforms Lark parse tree into AST nodes.
Uses Lark's Transformer pattern to recursively convert the tree.
"""

from lark import Transformer, v_args

from ast_nodes import (
    # Expressions
    Number, Var, BinOp,
    # Time system
    Event, SchedVar, RangeOffset, Time, Interval, EventBind,
    # Ports
    PortDef, PortRef, PortBundleAccess,
    # Module structure
    Param, Constraint, Signature,
    # Resource pools
    Pool, PoolSize, BindingIndex,
    # Commands
    Instance, Invocation, Connect, TimeLoop, SpaceLoop, IfStmt, ParamLet,
    BundleDecl,
    # Top-level
    Component, ExternBlock, Import, File,
)


class ASTBuilder(Transformer):
    """
    Transforms a Lark parse tree into an AST.

    Methods are named after grammar rules and receive the transformed
    children as arguments.
    """

    # ========================================================================
    # Terminals
    # ========================================================================

    def IDENT(self, token):
        return str(token)

    def NUMBER(self, token):
        return Number(int(token))

    def STRING(self, token):
        return str(token)[1:-1]  # Remove quotes

    def ORDER_OP(self, token):
        return str(token)

    def ADD_OP(self, token):
        return str(token)

    def MUL_OP(self, token):
        return str(token)

    # ========================================================================
    # File Structure
    # ========================================================================

    def start(self, items):
        imports = [i for i in items if isinstance(i, Import)]
        modules = [i for i in items if not isinstance(i, Import)]
        return File(imports=imports, modules=modules)

    def import_stmt(self, items):
        return Import(path=items[0])

    def module_def(self, items):
        return items[0]

    # ========================================================================
    # Time System: Events, SchedVars, Times
    # ========================================================================

    def event(self, items):
        """'G -> Event(name='G')"""
        return Event(name=items[0])

    # --- Scheduler Variables ---

    def sched_var_named(self, items):
        """?t_b1 -> SchedVar(name='t_b1')"""
        return SchedVar(name=items[0])

    def sched_var_anon(self, items):
        """? -> SchedVar(name=None)"""
        return SchedVar(name=None)

    # --- Range Offsets ---

    def range_bounded(self, items):
        """[lo..hi] -> RangeOffset(lo, hi)"""
        return RangeOffset(lo=items[0], hi=items[1])

    def range_lo_only(self, items):
        """[lo..] -> RangeOffset(lo, None)"""
        return RangeOffset(lo=items[0], hi=None)

    def range_hi_only(self, items):
        """[..hi] -> RangeOffset(None, hi)"""
        return RangeOffset(lo=None, hi=items[0])

    # --- Time Expressions (Event-based) ---

    def time_event(self, items):
        """'G -> Time(event=Event('G'))"""
        return Time(event=items[0])

    def time_offset(self, items):
        """'G+1 -> Time(event=Event('G'), offset=Number(1))"""
        return Time(event=items[0], offset=items[1])

    def time_event_range(self, items):
        """'G+[4..] -> Time(event=Event('G'), offset=RangeOffset(4, None))"""
        return Time(event=items[0], offset=items[1])

    # --- Time Expressions (SchedVar-based) ---

    def time_sched(self, items):
        """?t -> Time(sched_var=SchedVar('t'))"""
        return Time(sched_var=items[0])

    def time_sched_offset(self, items):
        """?t+1 -> Time(sched_var=SchedVar('t'), offset=Number(1))"""
        return Time(sched_var=items[0], offset=items[1])

    def time_sched_range(self, items):
        """?t+[2..4] -> Time(sched_var=SchedVar('t'), offset=RangeOffset(2,4))"""
        return Time(sched_var=items[0], offset=items[1])

    def time_named_bind(self, items):
        """?tloop='G -> inner Time with alias='tloop'
        items[0]: str (IDENT name), items[1]: Time (inner time expression)
        """
        import dataclasses
        name = items[0]   # str
        inner = items[1]  # Time
        return dataclasses.replace(inner, alias=name)

    def time_loop_end(self, items):
        """?alias.e -> Time(sched_var=SchedVar(alias), loop_end=True)
        items[0]: str (alias name)
        """
        name = str(items[0])
        return Time(sched_var=SchedVar(name=name), loop_end=True)

    def time_loop_end_offset(self, items):
        """?alias.e+N -> Time(sched_var=SchedVar(alias), loop_end=True, offset=N)
        items[0]: str (alias name), items[1]: offset expr
        """
        name = str(items[0])
        return Time(sched_var=SchedVar(name=name), loop_end=True, offset=items[1])

    # --- Anonymous Time (for intervals) ---

    def time_anon(self, items):
        """? in interval -> Time(sched_var=SchedVar(None))"""
        return Time(sched_var=SchedVar(name=None))

    def time_or_anon(self, items):
        return items[0]

    # --- Event Bindings ---

    def event_bind_expr(self, items):
        """'G: 1 -> EventBind(event, delay=Number(1))"""
        return EventBind(event=items[0], delay=items[1])

    def event_bind_sched(self, items):
        """'G: ? -> EventBind(event, delay=SchedVar(None))"""
        return EventBind(event=items[0], delay=items[1])

    def event_bind_no_ii(self, items):
        """'G -> EventBind(event, delay=None)  — non-pipelined, no II"""
        return EventBind(event=items[0], delay=None)

    def abstract_vars(self, items):
        return list(items)

    # ========================================================================
    # Intervals
    # ========================================================================

    def interval(self, items):
        """[start, end] -> Interval(start, end)"""
        return Interval(start=items[0], end=items[1])

    # ========================================================================
    # Parameters
    # ========================================================================

    def param(self, items):
        return Param(name=items[0])

    def params(self, items):
        return list(items)

    # ========================================================================
    # Ports and IO
    # ========================================================================

    def port_interface(self, items):
        """go: interface['G]"""
        return PortDef(
            name=items[0],
            interval=None,
            width=None,
            is_interface=True,
            interface_event=items[1]
        )

    def port_signal(self, items):
        """left: ['G, 'G+1] 32"""
        return PortDef(
            name=items[0],
            interval=items[1],
            width=items[2],
            is_interface=False
        )

    def port_bundle_signal(self, items):
        """a[8]: ['G, 'G+1] 32"""
        return PortDef(
            name=items[0],
            interval=items[2],
            width=items[3],
            is_interface=False,
            bundle_size=items[1].value,  # NUMBER -> Number node -> .value
        )

    def port_defs(self, items):
        return list(items)

    def input_ports(self, items):
        """Wrapper for input port_defs"""
        return items[0] if items else []

    def output_ports(self, items):
        """Wrapper for output port_defs"""
        return items[0] if items else []

    def io(self, items):
        """(inputs) -> (outputs) returns tuple"""
        inputs = items[0] if len(items) > 0 else []
        outputs = items[1] if len(items) > 1 else []
        return (inputs, outputs)

    # ========================================================================
    # Constraints
    # ========================================================================

    def constraint(self, items):
        """W > 0 -> Constraint(left, op, right)"""
        return Constraint(left=items[0], op=items[1], right=items[2])

    def constraints(self, items):
        return list(items)

    def expr_cmp(self, items):
        """Comparison in if statements"""
        return Constraint(left=items[0], op=items[1], right=items[2])

    # ========================================================================
    # Module Definitions
    # ========================================================================

    def comp_module(self, items):
        """comp Name[params]<events>(io) where constraints { commands }"""
        name, params, event_binds, inputs, outputs, constraints, commands = \
            self._parse_signature_items(items, with_commands=True)

        sig = Signature(
            name=name,
            params=params,
            event_binds=event_binds,
            inputs=inputs,
            outputs=outputs,
            constraints=constraints
        )
        return Component(signature=sig, commands=commands)

    def signature(self, items):
        """Signature in extern block (no body)"""
        name, params, event_binds, inputs, outputs, constraints, _ = \
            self._parse_signature_items(items, with_commands=False)

        return Signature(
            name=name,
            params=params,
            event_binds=event_binds,
            inputs=inputs,
            outputs=outputs,
            constraints=constraints
        )

    def _parse_signature_items(self, items, with_commands=False):
        """Helper to parse signature components from mixed item list."""
        name = ""
        params = []
        event_binds = []
        inputs = []
        outputs = []
        constraints = []
        commands = []

        for item in items:
            if isinstance(item, str):
                name = item
            elif isinstance(item, list) and len(item) > 0:
                first = item[0]
                if isinstance(first, Param):
                    params = item
                elif isinstance(first, EventBind):
                    event_binds = item
                elif isinstance(first, Constraint):
                    constraints = item
            elif isinstance(item, tuple) and len(item) == 2:
                inputs, outputs = item
            elif with_commands and isinstance(item, (Pool, Instance, Invocation, Connect, TimeLoop, SpaceLoop, IfStmt, ParamLet, BundleDecl)):
                commands.append(item)

        return name, params, event_binds, inputs, outputs, constraints, commands

    def extern_block(self, items):
        """extern "path" { signatures }"""
        path = items[0]
        signatures = [i for i in items[1:] if isinstance(i, Signature)]
        return ExternBlock(path=path, signatures=signatures)

    # ========================================================================
    # Resource Pools
    # ========================================================================

    def pool_decl(self, items):
        """pool A : Add[32] * N;"""
        name = items[0]
        module = items[1]
        params = []
        size = None

        for item in items[2:]:
            if isinstance(item, list):
                params = item
            elif isinstance(item, PoolSize):
                size = item

        return Pool(name=name, module=module, params=params, size=size)

    def pool_max(self, items):
        """N -> PoolSize with max_count"""
        return PoolSize(max_count=items[0], is_unbounded=False)

    def pool_unbounded(self, items):
        """? -> PoolSize unbounded"""
        return PoolSize(max_count=None, is_unbounded=True)

    # ========================================================================
    # Binding Index
    # ========================================================================

    def binding_index(self, items):
        """[bind_idx] -> BindingIndex"""
        return items[0]

    def bind_anon(self, items):
        """[?] -> anonymous binding"""
        return BindingIndex(kind='anon', value=None)

    def bind_named(self, items):
        """[?x] -> named binding var"""
        return BindingIndex(kind='named', value=items[0])

    def bind_explicit(self, items):
        """[0] -> explicit index"""
        return BindingIndex(kind='explicit', value=items[0])

    # ========================================================================
    # Commands
    # ========================================================================

    def command(self, items):
        return items[0]

    def instance(self, items):
        """A := new Add[32];"""
        name = items[0]
        module = items[1]
        params = []
        time_args = []
        args = []

        for item in items[2:]:
            if isinstance(item, list):
                if item and isinstance(item[0], Time):
                    time_args = item
                elif item and isinstance(item[0], Number):
                    params = item
                elif item and all(isinstance(x, (Number, Var, BinOp)) for x in item):
                    params = item
            elif isinstance(item, tuple):
                time_args, args = item

        return Instance(name=name, module=module, params=params, time_args=time_args, args=args)

    def conc_params(self, items):
        """[32] in new Add[32]"""
        return list(items)

    def invoke_args(self, items):
        """<'G>(a, b) -> (times, args)"""
        times = []
        args = []
        for item in items:
            if isinstance(item, Time):
                times.append(item)
            elif isinstance(item, list):
                if item and isinstance(item[0], Time):
                    times.extend(item)
                else:
                    args = item
        return (times, args)

    def arguments(self, items):
        return list(items)

    def invocation(self, items):
        """result := A[?]<'G>(a, b); or result := A<'G>(a, b);"""
        name = items[0]
        instance = items[1]
        binding = None
        time_args = []
        args = []

        for item in items[2:]:
            if isinstance(item, BindingIndex):
                binding = item
            elif isinstance(item, tuple):
                time_args, args = item

        return Invocation(name=name, instance=instance, binding=binding, time_args=time_args, args=args)

    def connect(self, items):
        """out = result.out;"""
        return Connect(dest=items[0], src=items[1])

    # ========================================================================
    # Time Loop
    # ========================================================================

    def timeloop_start(self, items):
        """<?start> -> Time"""
        return items[0]

    def ii_anon(self, items):
        """[II=?] -> SchedVar(None)"""
        return SchedVar(name=None)

    def ii_expr(self, items):
        """[II=4] -> Number(4) (or any expr)"""
        return items[0]

    def timeloop_ii(self, items):
        """[II=val] -> unwrap the ii_val (Number or SchedVar)"""
        return items[0]  # ii_anon or ii_expr result

    def timeloop(self, items):
        """timeloop<start> [II=?] for i in 0..N { ... }"""
        # Items arrive as: Time [ii?] IDENT lower_bound upper_bound command*
        # start is now required; lower_bound is discarded.
        idx = 0
        start = items[idx]; idx += 1
        ii = None
        if idx < len(items) and isinstance(items[idx], (Number, SchedVar)):
            ii = items[idx]
            idx += 1
        var = items[idx]; idx += 1
        # lower bound (always 0, discard)
        idx += 1
        end = items[idx]; idx += 1
        body = list(items[idx:])

        return TimeLoop(var=var, start=start, ii=ii, end=end, body=body)

    # ========================================================================
    # Space Loop
    # ========================================================================

    def spaceloop(self, items):
        """spaceloop<start> for i in start..end { ... }"""
        # Items arrive as: Time IDENT lower_bound upper_bound command*
        # start is now required; lower_bound is discarded.
        idx = 0
        start = items[idx]; idx += 1
        var = items[idx]; idx += 1
        # lower bound (discard)
        idx += 1
        end = items[idx]; idx += 1
        body = list(items[idx:])
        return SpaceLoop(var=var, start=start, end=end, body=body)

    def if_stmt(self, items):
        """if cond { ... } else { ... }"""
        condition = items[0]
        then_body = []
        else_body = None

        for item in items[1:]:
            if isinstance(item, (Pool, Instance, Invocation, Connect, TimeLoop, SpaceLoop, IfStmt, ParamLet, BundleDecl)):
                then_body.append(item)

        return IfStmt(condition=condition, then_body=then_body, else_body=else_body)

    def param_let(self, items):
        """let x = expr;"""
        return ParamLet(name=items[0], value=items[1])

    def bundle_decl(self, items):
        """bundle twiddle[2]: ['G, 'G+1] 32;"""
        return BundleDecl(
            name=items[0],
            bundle_size=items[1].value,  # NUMBER -> Number node -> .value
            interval=items[2],
            width=items[3],
        )

    # ========================================================================
    # Port Expressions
    # ========================================================================

    def port_qualified(self, items):
        """instance.port"""
        return PortRef(instance=items[0], port=items[1])

    def port_simple(self, items):
        """port (no instance qualifier)"""
        return PortRef(instance=None, port=items[0])

    def port_literal(self, items):
        """Numeric literal as port value"""
        return items[0]

    def port_bundle_simple(self, items):
        """bundle{i}"""
        return PortBundleAccess(instance=None, bundle=items[0], index=items[1])

    def port_bundle_qualified(self, items):
        """inst.bundle{i}"""
        return PortBundleAccess(instance=items[0], bundle=items[1], index=items[2])

    # ========================================================================
    # Expressions
    # ========================================================================

    def expr_ident(self, items):
        return Var(name=items[0])

    def expr_number(self, items):
        return items[0]

    @v_args(inline=False)
    def expr(self, items):
        """Build left-associative + - expressions"""
        if len(items) == 1:
            return items[0]
        result = items[0]
        for i in range(1, len(items), 2):
            result = BinOp(op=str(items[i]), left=result, right=items[i + 1])
        return result

    @v_args(inline=False)
    def term(self, items):
        """Build left-associative * / % expressions"""
        if len(items) == 1:
            return items[0]
        result = items[0]
        for i in range(1, len(items), 2):
            result = BinOp(op=str(items[i]), left=result, right=items[i + 1])
        return result

    def factor(self, items):
        return items[0]


def build_ast(parse_tree):
    """Convert a Lark parse tree to an AST."""
    return ASTBuilder().transform(parse_tree)
