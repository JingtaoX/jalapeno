"""
Grammar Tests for Jalapeno Parser

Tests each grammar rule with focused examples to ensure correct parsing.
Run with: python -m pytest tests/test_grammar.py -v
Or: python tests/test_grammar.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser import create_parser
from ast_builder import build_ast
from ast_nodes import (
    Number, Var, BinOp,
    Event, SchedVar, RangeOffset, Time, Interval, EventBind,
    PortDef, PortRef, PortBundleAccess,
    Param, Constraint, Signature,
    Pool, PoolSize, BindingIndex,
    Instance, Invocation, Connect, TimeLoop, SpaceLoop, IfStmt, ParamLet,
    BundleDecl,
    Component, ExternBlock, Import, File,
)


class TestGrammar:
    """Test suite for Jalapeno grammar rules."""

    def setup_method(self):
        self.parser = create_parser()

    def parse(self, code):
        """Parse code and return AST."""
        tree = self.parser.parse(code)
        return build_ast(tree)

    # ========================================================================
    # File Structure Tests
    # ========================================================================

    def test_empty_file(self):
        """Empty file should parse to empty File."""
        ast = self.parse("")
        assert isinstance(ast, File)
        assert ast.imports == []
        assert ast.modules == []

    def test_import_statement(self):
        """import "path/to/file.fil";"""
        ast = self.parse('import "primitives/core.fil";')
        assert len(ast.imports) == 1
        assert ast.imports[0].path == "primitives/core.fil"

    def test_multiple_imports(self):
        """Multiple imports."""
        ast = self.parse('''
            import "a.fil";
            import "b.fil";
        ''')
        assert len(ast.imports) == 2

    # ========================================================================
    # Event and Time Tests
    # ========================================================================

    def test_event_simple(self):
        """'G event in abstract vars."""
        ast = self.parse("comp Foo<'G: 1>() -> () {}")
        event_bind = ast.modules[0].signature.event_binds[0]
        assert event_bind.event.name == "G"
        assert isinstance(event_bind.delay, Number)
        assert event_bind.delay.value == 1

    def test_event_bind_with_sched_var(self):
        """'G: ? - event bound to anonymous scheduler var."""
        ast = self.parse("comp Foo<'G: ?>() -> () {}")
        event_bind = ast.modules[0].signature.event_binds[0]
        assert isinstance(event_bind.delay, SchedVar)
        assert event_bind.delay.name is None

    def test_sched_var_named(self):
        """?t_b1 - named scheduler variable."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t_b1>(a);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert isinstance(inv.time_args[0].sched_var, SchedVar)
        assert inv.time_args[0].sched_var.name == "t_b1"

    def test_sched_var_anonymous(self):
        """? - anonymous scheduler variable."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?>( a);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.time_args[0].sched_var.name is None

    def test_time_event_only(self):
        """'G - just event, no offset."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<'G>(a);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.time_args[0].event.name == "G"
        assert inv.time_args[0].offset is None

    def test_time_event_with_offset(self):
        """'G+1 - event with expression offset."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<'G+1>(a);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.time_args[0].event.name == "G"
        assert inv.time_args[0].offset.value == 1

    def test_time_sched_with_offset(self):
        """?t+1 - scheduler var with expression offset."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t+1>(a);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.time_args[0].sched_var.name == "t"
        assert inv.time_args[0].offset.value == 1

    # ========================================================================
    # Range Offset Tests
    # ========================================================================

    def test_range_bounded(self):
        """[2..4] - both bounds specified."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t+[2..4]>(a);
            }
        ''')
        inv = ast.modules[0].commands[1]
        offset = inv.time_args[0].offset
        assert isinstance(offset, RangeOffset)
        assert offset.lo.value == 2
        assert offset.hi.value == 4

    def test_range_lo_only(self):
        """[4..] - lower bound only."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<'G+[4..]>(a);
            }
        ''')
        inv = ast.modules[0].commands[1]
        offset = inv.time_args[0].offset
        assert isinstance(offset, RangeOffset)
        assert offset.lo.value == 4
        assert offset.hi is None

    def test_range_hi_only(self):
        """[..4] - upper bound only."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t+[..4]>(a);
            }
        ''')
        inv = ast.modules[0].commands[1]
        offset = inv.time_args[0].offset
        assert isinstance(offset, RangeOffset)
        assert offset.lo is None
        assert offset.hi.value == 4

    # ========================================================================
    # Interval Tests
    # ========================================================================

    def test_interval_with_events(self):
        """['G, 'G+1] - standard interval."""
        ast = self.parse('''
            comp Foo<'G: 1>(
                x: ['G, 'G+1] 32
            ) -> () {}
        ''')
        port = ast.modules[0].signature.inputs[0]
        assert port.interval.start.event.name == "G"
        assert port.interval.end.event.name == "G"
        assert port.interval.end.offset.value == 1

    def test_interval_with_anonymous(self):
        """[?, ?] - anonymous scheduler vars for output."""
        ast = self.parse('''
            comp Foo<'G: ?>() -> (
                out: [?, ?] 32
            ) {}
        ''')
        port = ast.modules[0].signature.outputs[0]
        assert port.interval.start.sched_var.name is None
        assert port.interval.end.sched_var.name is None

    # ========================================================================
    # Port Definition Tests
    # ========================================================================

    def test_port_signal(self):
        """Standard signal port: name: [interval] width"""
        ast = self.parse('''
            comp Foo<'G: 1>(
                data: ['G, 'G+1] 32
            ) -> () {}
        ''')
        port = ast.modules[0].signature.inputs[0]
        assert port.name == "data"
        assert port.width.value == 32
        assert not port.is_interface

    def test_port_interface(self):
        """Interface port: name: interface[event]"""
        ast = self.parse('''
            comp Foo<'G: 1>(
                go: interface['G]
            ) -> () {}
        ''')
        port = ast.modules[0].signature.inputs[0]
        assert port.name == "go"
        assert port.is_interface
        assert port.interface_event.name == "G"

    # ========================================================================
    # Parameter Tests
    # ========================================================================

    def test_component_params(self):
        """comp Foo[W, N]<...>"""
        ast = self.parse("comp Foo[W, N]<'G: 1>() -> () {}")
        params = ast.modules[0].signature.params
        assert len(params) == 2
        assert params[0].name == "W"
        assert params[1].name == "N"

    def test_component_no_params(self):
        """comp Foo<...> - no params."""
        ast = self.parse("comp Foo<'G: 1>() -> () {}")
        params = ast.modules[0].signature.params
        assert len(params) == 0

    # ========================================================================
    # Constraint Tests
    # ========================================================================

    def test_constraint_gt(self):
        """where W > 0"""
        ast = self.parse("comp Foo[W]<'G: 1>() -> () where W > 0 {}")
        constraints = ast.modules[0].signature.constraints
        assert len(constraints) == 1
        assert constraints[0].op == ">"
        assert constraints[0].left.name == "W"
        assert constraints[0].right.value == 0

    def test_constraint_multiple(self):
        """where W > 0, N >= 1"""
        ast = self.parse("comp Foo[W, N]<'G: 1>() -> () where W > 0, N >= 1 {}")
        constraints = ast.modules[0].signature.constraints
        assert len(constraints) == 2
        assert constraints[0].op == ">"
        assert constraints[1].op == ">="

    def test_constraint_all_ops(self):
        """Test all comparison operators."""
        ops = [">", "<", ">=", "<=", "=="]
        for op in ops:
            ast = self.parse(f"comp Foo[W]<'G: 1>() -> () where W {op} 0 {{}}")
            assert ast.modules[0].signature.constraints[0].op == op

    # ========================================================================
    # Command Tests
    # ========================================================================

    def test_instance_simple(self):
        """A := new Add[32];"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, Instance)
        assert cmd.name == "A"
        assert cmd.module == "Add"
        assert cmd.params[0].value == 32

    def test_instance_no_params(self):
        """A := new Add;"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert cmd.params == []

    def test_invocation(self):
        """result := A<'G>(a, b);"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                result := A<'G>(a, b);
            }
        ''')
        cmd = ast.modules[0].commands[1]
        assert isinstance(cmd, Invocation)
        assert cmd.name == "result"
        assert cmd.instance == "A"
        assert len(cmd.args) == 2

    def test_connect(self):
        """out = result.out;"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                out = src.data;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, Connect)
        assert cmd.dest.port == "out"
        assert cmd.src.instance == "src"
        assert cmd.src.port == "data"

    # ========================================================================
    # Time Loop Tests
    # ========================================================================

    def test_timeloop_minimal(self):
        """timeloop<'G> [II=?] for i in 0..N { ... } — start required"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                pool M : Mult[32] * 1;
                timeloop<'G> [II=?] for i in 0..N {
                    r := M[?]<?t>(a, b);
                }
            }
        ''')
        cmd = ast.modules[0].commands[1]
        assert isinstance(cmd, TimeLoop)
        assert cmd.var == "i"
        assert cmd.start is not None
        assert cmd.start.event.name == "G"
        assert isinstance(cmd.ii, SchedVar)
        assert cmd.ii.name is None  # anonymous => solver decides
        assert cmd.end.name == "N"
        assert len(cmd.body) == 1

    def test_timeloop_fixed_ii(self):
        """timeloop<'G> [II=4] for i in 0..N { ... } — user-fixed II"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                pool M : Mult[32] * 1;
                timeloop<'G> [II=4] for i in 0..N {
                    r := M[?]<?t>(a, b);
                }
            }
        ''')
        cmd = ast.modules[0].commands[1]
        assert isinstance(cmd, TimeLoop)
        assert isinstance(cmd.ii, Number)
        assert cmd.ii.value == 4

    def test_timeloop_sched_start(self):
        """timeloop<?s> [II=?] for i in 0..N { ... } — named sched var start"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                pool M : Mult[32] * 1;
                timeloop<?s> [II=?] for i in 0..N {
                    r := M[?]<?t>(a, b);
                }
            }
        ''')
        cmd = ast.modules[0].commands[1]
        assert isinstance(cmd, TimeLoop)
        assert cmd.start is not None
        assert cmd.start.sched_var is not None
        assert cmd.start.sched_var.name == "s"
        assert cmd.start.offset is None

    def test_timeloop_event_start(self):
        """timeloop<'G> [II=4] for i in 0..N { ... } — event-pinned start"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                pool M : Mult[32] * 1;
                timeloop<'G> [II=4] for i in 0..N {
                    r := M[?]<?t>(a, b);
                }
            }
        ''')
        cmd = ast.modules[0].commands[1]
        assert isinstance(cmd, TimeLoop)
        assert cmd.start.event.name == "G"
        assert cmd.start.offset is None

    def test_timeloop_event_offset_start(self):
        """timeloop<'G+2> [II=4] for i in 0..N { ... }"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                pool M : Mult[32] * 1;
                timeloop<'G+2> [II=4] for i in 0..N {
                    r := M[?]<?t>(a, b);
                }
            }
        ''')
        cmd = ast.modules[0].commands[1]
        assert isinstance(cmd, TimeLoop)
        assert cmd.start.event.name == "G"
        assert cmd.start.offset.value == 2

    def test_timeloop_no_ii_clause(self):
        """timeloop<'G> for i in 0..N { ... } — start required, II omitted"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                pool M : Mult[32] * 1;
                timeloop<'G> for i in 0..N {
                    r := M[?]<?t>(a, b);
                }
            }
        ''')
        cmd = ast.modules[0].commands[1]
        assert isinstance(cmd, TimeLoop)
        assert cmd.start is not None
        assert cmd.start.event.name == "G"
        assert cmd.ii is None

    def test_timeloop_body_commands(self):
        """Body can contain pool decls, invocations, connects."""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                timeloop<'G> [II=?] for i in 0..N {
                    pool M : Mult[32] * 1;
                    r := M[?u]<?t>(a, b);
                    out = r.out;
                }
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, TimeLoop)
        assert len(cmd.body) == 3
        assert isinstance(cmd.body[0], Pool)
        assert isinstance(cmd.body[1], Invocation)
        assert isinstance(cmd.body[2], Connect)

    def test_timeloop_named_binding_in_body(self):
        """Named binding and named timing inside a timeloop."""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                pool M : Mult[32] * 2;
                timeloop<?s> [II=?] for i in 0..N {
                    m1 := M[?u]<?t>(a, b);
                    m2 := M[?u]<?t+3>(c, d);
                }
            }
        ''')
        cmd = ast.modules[0].commands[1]
        body = cmd.body
        assert body[0].binding.kind == 'named'
        assert body[0].binding.value == 'u'
        assert body[1].binding.kind == 'named'
        assert body[1].binding.value == 'u'
        assert body[1].time_args[0].offset.value == 3

    # ========================================================================
    # Space Loop Tests
    # ========================================================================

    def test_spaceloop_basic(self):
        """spaceloop<'G> for i in 0..N { ... }"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                spaceloop<'G> for i in 0..N {
                    A := new Add[32];
                    r := A<'G>(x, y);
                }
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, SpaceLoop)
        assert cmd.var == "i"
        assert cmd.start.event.name == "G"
        assert cmd.end.name == "N"
        assert len(cmd.body) == 2

    def test_spaceloop_expr_bounds(self):
        """spaceloop<'G> for i in 0..N*2 { ... }"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                spaceloop<'G> for i in 0..N*2 {
                    out = x;
                }
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, SpaceLoop)
        assert isinstance(cmd.end, BinOp)
        assert cmd.end.op == "*"

    def test_param_let(self):
        """let x = expr;"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                let x = 42;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, ParamLet)
        assert cmd.name == "x"
        assert cmd.value.value == 42

    def test_if_stmt(self):
        """if cond { ... }"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                if N > 0 {
                    out = x;
                }
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, IfStmt)
        assert cmd.condition.op == ">"
        assert len(cmd.then_body) == 1

    # ========================================================================
    # Expression Tests
    # ========================================================================

    def test_expr_number(self):
        """Numeric literal."""
        ast = self.parse("comp Foo<'G: 42>() -> () {}")
        assert ast.modules[0].signature.event_binds[0].delay.value == 42

    def test_expr_ident(self):
        """Variable reference."""
        ast = self.parse("comp Foo[W]<'G: W>() -> () {}")
        assert ast.modules[0].signature.event_binds[0].delay.name == "W"

    def test_expr_binop_add(self):
        """W + 1"""
        ast = self.parse("comp Foo[W]<'G: 1>() -> () where W + 1 > 0 {}")
        left = ast.modules[0].signature.constraints[0].left
        assert isinstance(left, BinOp)
        assert left.op == "+"

    def test_expr_binop_mul(self):
        """W * 2"""
        ast = self.parse("comp Foo[W]<'G: 1>() -> () where W * 2 > 0 {}")
        left = ast.modules[0].signature.constraints[0].left
        assert isinstance(left, BinOp)
        assert left.op == "*"

    def test_expr_complex(self):
        """W * 2 + 1 - precedence test."""
        ast = self.parse("comp Foo[W]<'G: 1>() -> () where W * 2 + 1 > 0 {}")
        left = ast.modules[0].signature.constraints[0].left
        # Should be (W * 2) + 1
        assert left.op == "+"
        assert left.left.op == "*"

    # ========================================================================
    # Port Expression Tests
    # ========================================================================

    def test_port_ref_simple(self):
        """Simple port reference: x"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                out = x;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert cmd.src.instance is None
        assert cmd.src.port == "x"

    def test_port_ref_qualified(self):
        """Qualified port reference: inst.port"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                out = inst.data;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert cmd.src.instance == "inst"
        assert cmd.src.port == "data"

    # ========================================================================
    # Extern Block Tests
    # ========================================================================

    def test_extern_block(self):
        """extern "path" { signatures }"""
        ast = self.parse('''
            extern "primitives/core.fil" {
                comp Add[W]<'G: 1>(
                    left: ['G, 'G+1] W,
                    right: ['G, 'G+1] W
                ) -> (
                    out: ['G, 'G+1] W
                );
            }
        ''')
        ext = ast.modules[0]
        assert isinstance(ext, ExternBlock)
        assert ext.path == "primitives/core.fil"
        assert len(ext.signatures) == 1
        assert ext.signatures[0].name == "Add"

    # ========================================================================
    # Comment Tests
    # ========================================================================

    def test_line_comment(self):
        """// comment"""
        ast = self.parse('''
            // This is a comment
            comp Foo<'G: 1>() -> () {}
        ''')
        assert len(ast.modules) == 1

    def test_block_comment(self):
        """/* comment */"""
        ast = self.parse('''
            /* Multi-line
               comment */
            comp Foo<'G: 1>() -> () {}
        ''')
        assert len(ast.modules) == 1

    # ========================================================================
    # Pool Declaration Tests
    # ========================================================================

    def test_pool_max(self):
        """pool A : Add[32] * 2;"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                pool Adders : Add[32] * 2;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, Pool)
        assert cmd.name == "Adders"
        assert cmd.module == "Add"
        assert cmd.params[0].value == 32
        assert cmd.size.max_count.value == 2
        assert cmd.size.is_unbounded is False

    def test_pool_unbounded(self):
        """pool A : Add[32] * ?;"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                pool Adders : Add[32] * ?;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, Pool)
        assert cmd.size.max_count is None
        assert cmd.size.is_unbounded is True

    def test_pool_no_params(self):
        """pool A : Add * 2;"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                pool A : SimpleMod * 1;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert cmd.params == []

    # ========================================================================
    # Binding Index Tests
    # ========================================================================

    def test_binding_anon(self):
        """A[?] - anonymous binding"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                pool A : Add[32] * 2;
                x := A[?]<?t>(a, b);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert isinstance(inv, Invocation)
        assert inv.binding is not None
        assert inv.binding.kind == 'anon'
        assert inv.binding.value is None

    def test_binding_named(self):
        """A[?x] - named binding var"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                pool A : Add[32] * 2;
                x := A[?idx]<?t>(a, b);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.binding.kind == 'named'
        assert inv.binding.value == 'idx'

    def test_binding_explicit(self):
        """A[0] - explicit index"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                pool A : Add[32] * 2;
                x := A[0]<?t>(a, b);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.binding.kind == 'explicit'
        assert inv.binding.value.value == 0

    def test_binding_explicit_expr(self):
        """A[N-1] - explicit index with expression"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                pool A : Add[32] * N;
                x := A[N-1]<?t>(a, b);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.binding.kind == 'explicit'
        assert isinstance(inv.binding.value, BinOp)
        assert inv.binding.value.op == '-'

    def test_invocation_no_binding(self):
        """Traditional invocation without binding"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<'G>(a, b);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.binding is None

    def test_multiple_bindings_same_name(self):
        """Multiple invocations with same binding var"""
        ast = self.parse('''
            comp Foo<'G: ?>() -> () {
                pool A : Add[32] * 2;
                x := A[?i]<?t1>(a, b);
                y := A[?i]<?t2>(c, d);
            }
        ''')
        inv1 = ast.modules[0].commands[1]
        inv2 = ast.modules[0].commands[2]
        assert inv1.binding.kind == 'named'
        assert inv1.binding.value == 'i'
        assert inv2.binding.kind == 'named'
        assert inv2.binding.value == 'i'

    def test_mixed_bindings(self):
        """Mix of anonymous and explicit bindings"""
        ast = self.parse('''
            comp Foo<'G: ?>() -> () {
                pool A : Add[32] * 3;
                x := A[?]<?t1>(a, b);
                y := A[0]<?t2>(c, d);
                z := A[?j]<?t3>(e, f);
            }
        ''')
        cmds = ast.modules[0].commands
        assert cmds[1].binding.kind == 'anon'
        assert cmds[2].binding.kind == 'explicit'
        assert cmds[3].binding.kind == 'named'

    # ========================================================================
    # Bundle Tests
    # ========================================================================

    def test_port_bundle_signal(self):
        """a[8]: ['G, 'G+1] 32  — bundle port in signature."""
        ast = self.parse('''
            comp Foo<'G: 1>(
                a[8]: ['G, 'G+1] 32,
            ) -> () {}
        ''')
        port = ast.modules[0].signature.inputs[0]
        assert isinstance(port, PortDef)
        assert port.name == "a"
        assert port.bundle_size == 8
        assert port.width.value == 32

    def test_port_bundle_signal_output(self):
        """Bundle port in output signature."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> (
                out[4]: ['G, 'G+1] 32,
            ) {}
        ''')
        port = ast.modules[0].signature.outputs[0]
        assert isinstance(port, PortDef)
        assert port.name == "out"
        assert port.bundle_size == 4

    def test_scalar_port_bundle_size_none(self):
        """Scalar port has bundle_size=None."""
        ast = self.parse('''
            comp Foo<'G: 1>(
                x: ['G, 'G+1] 32,
            ) -> () {}
        ''')
        port = ast.modules[0].signature.inputs[0]
        assert port.bundle_size is None

    def test_bundle_decl_command(self):
        """bundle twiddle[2]: ['G, 'G+1] 32;"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                bundle twiddle[2]: ['G, 'G+1] 32;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, BundleDecl)
        assert cmd.name == "twiddle"
        assert cmd.bundle_size == 2
        assert cmd.width.value == 32

    def test_bundle_decl_interval(self):
        """bundle decl captures interval correctly."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                bundle buf[4]: ['G, 'G+1] 16;
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, BundleDecl)
        assert cmd.interval.start.event.name == "G"
        assert cmd.interval.end.event.name == "G"
        assert cmd.interval.end.offset.value == 1

    def test_port_bundle_access_simple(self):
        """a{i} — simple bundle element access in port_expr."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                out = a{0};
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd.src, PortBundleAccess)
        assert cmd.src.instance is None
        assert cmd.src.bundle == "a"
        assert cmd.src.index.value == 0

    def test_port_bundle_access_qualified(self):
        """inst.buf{i} — qualified bundle element access."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                out = inst.buf{2};
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd.src, PortBundleAccess)
        assert cmd.src.instance == "inst"
        assert cmd.src.bundle == "buf"
        assert cmd.src.index.value == 2

    def test_port_bundle_access_var_index(self):
        """a{j} — variable index in bundle access."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                out = a{j};
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd.src, PortBundleAccess)
        assert isinstance(cmd.src.index, Var)
        assert cmd.src.index.name == "j"

    def test_bundle_in_invocation_args(self):
        """Bundle access used as invocation argument."""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                pool A : Add[32] * 1;
                r := A[?]<?t>(a{0}, b{1});
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert isinstance(inv.args[0], PortBundleAccess)
        assert inv.args[0].bundle == "a"
        assert inv.args[0].index.value == 0
        assert isinstance(inv.args[1], PortBundleAccess)
        assert inv.args[1].bundle == "b"
        assert inv.args[1].index.value == 1

    def test_bundle_port_and_decl_coexist(self):
        """Bundle port in signature + bundle decl in body."""
        ast = self.parse('''
            comp Foo<'G: 1>(
                a[4]: ['G, 'G+1] 32,
            ) -> () {
                bundle tmp[4]: ['G, 'G+1] 32;
            }
        ''')
        port = ast.modules[0].signature.inputs[0]
        assert port.bundle_size == 4
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, BundleDecl)
        assert cmd.bundle_size == 4

    # ========================================================================
    # Named Time Bind Tests
    # ========================================================================

    def test_time_named_bind_on_event(self):
        """?tloop='G -> Time with event='G' and alias='tloop'"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                pool M : Mult[32] * 1;
                timeloop<?tloop='G> [II=?] for i in 0..N {
                    r := M[?]<?t>(a, b);
                }
            }
        ''')
        cmd = ast.modules[0].commands[1]
        assert isinstance(cmd, TimeLoop)
        assert cmd.start.event.name == "G"
        assert cmd.start.alias == "tloop"
        assert cmd.start.sched_var is None

    def test_time_named_bind_on_event_offset(self):
        """?t='G+1 -> Time(event='G', offset=1, alias='t')"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t='G+1>(a, b);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.time_args[0].event.name == "G"
        assert inv.time_args[0].offset.value == 1
        assert inv.time_args[0].alias == "t"

    def test_time_named_bind_on_sched_var(self):
        """?alias=?base -> Time(sched_var='base', alias='alias')"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?myname=?s>(a, b);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.time_args[0].sched_var.name == "s"
        assert inv.time_args[0].alias == "myname"

    def test_time_named_bind_in_spaceloop_start(self):
        """?sl='G in spaceloop start"""
        ast = self.parse('''
            comp Foo[N]<'G: 1>() -> () {
                spaceloop<?sl='G> for i in 0..N {
                    out = x;
                }
            }
        ''')
        cmd = ast.modules[0].commands[0]
        assert isinstance(cmd, SpaceLoop)
        assert cmd.start.event.name == "G"
        assert cmd.start.alias == "sl"

    def test_plain_time_has_alias_none(self):
        """Plain ?t has alias=None"""
        ast = self.parse('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t>(a, b);
            }
        ''')
        inv = ast.modules[0].commands[1]
        assert inv.time_args[0].alias is None
        assert inv.time_args[0].sched_var.name == "t"


# ============================================================================
# Run tests directly
# ============================================================================

def run_tests():
    """Run all tests and report results."""
    import traceback

    test_class = TestGrammar()
    test_class.setup_method()

    # Get all test methods
    test_methods = [m for m in dir(test_class) if m.startswith('test_')]

    passed = 0
    failed = 0
    errors = []

    for method_name in sorted(test_methods):
        method = getattr(test_class, method_name)
        try:
            method()
            print(f"  ✓ {method_name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {method_name}: {e}")
            failed += 1
            errors.append((method_name, traceback.format_exc()))
        except Exception as e:
            print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
            failed += 1
            errors.append((method_name, traceback.format_exc()))

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")

    if errors:
        print(f"\n{'='*60}")
        print("Failures:\n")
        for name, tb in errors:
            print(f"--- {name} ---")
            print(tb)

    return failed == 0


if __name__ == "__main__":
    print("Running Jalapeno Grammar Tests\n")
    success = run_tests()
    sys.exit(0 if success else 1)
