"""
Tests for the Scheduling IR
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser import create_parser
from ast_builder import build_ast
from ir import build_ir, SchedulingIR, Operation, Resource, LoopRegion, BundleInfo


class TestIR:
    """Test suite for IR building."""

    def setup_method(self):
        self.parser = create_parser()

    def parse_and_build_ir(self, code: str, module_latencies=None) -> SchedulingIR:
        tree = self.parser.parse(code)
        ast = build_ast(tree)
        component = ast.modules[0]
        return build_ir(component, module_latencies)

    # ========================================================================
    # Basic IR Building
    # ========================================================================

    def test_simple_invocation(self):
        """Single operation extracts correctly."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t1>(a, b);
            }
        ''')

        assert ir.name == "Foo"
        assert ir.start_event == "G"
        assert ir.ii == 1  # II (initiation interval)

        # Instance creates module latency entry, not a pool
        assert "A" in ir.module_latencies
        assert ir.module_latencies["A"] == 1
        assert "A" not in ir.pools  # Not a pool

        # Check operation - now keyed by resource.index
        assert "A.0" in ir.operations
        op = ir.operations["A.0"]
        assert op.op_name == "A.0"
        assert op.resource == "A"
        assert op.result == "x"  # Result wire name
        assert op.timing.sched_var == "t1"
        assert "a" in op.inputs
        assert "b" in op.inputs

        # Check result_to_op mapping
        assert ir.result_to_op["x"] == "A.0"

        # Check sched_vars collected
        assert "t1" in ir.sched_vars

    def test_operation_inputs(self):
        """Operation inputs are captured correctly."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t1>(a, b);
                y := A<?t2>(x.out, c);
            }
        ''')

        # Operations are now A.0 (x) and A.1 (y)
        assert "A.0" in ir.operations  # x
        assert "A.1" in ir.operations  # y

        # A.1 (y) has x.out as input
        assert "x.out" in ir.operations["A.1"].inputs
        assert "c" in ir.operations["A.1"].inputs

        # A.0 (x) has a, b as inputs
        assert "a" in ir.operations["A.0"].inputs
        assert "b" in ir.operations["A.0"].inputs

        # Check result mappings
        assert ir.result_to_op["x"] == "A.0"
        assert ir.result_to_op["y"] == "A.1"

    def test_three_add_pattern(self):
        """Diamond dependency pattern (like three_add)."""
        ir = self.parse_and_build_ir('''
            comp three_adds<'G: ?>() -> () {
                A := new Add[32];
                b1 := A<?t_b1>(a1, a2);
                b2 := A<?t_b2>(a3, a4);
                o  := A<?t_b3>(b1.out, b2.out);
            }
        ''')

        # Check all operations exist - now keyed by A.0, A.1, A.2
        assert set(ir.operations.keys()) == {"A.0", "A.1", "A.2"}

        # Check result mappings
        assert ir.result_to_op["b1"] == "A.0"
        assert ir.result_to_op["b2"] == "A.1"
        assert ir.result_to_op["o"] == "A.2"

        # Check inputs (dependencies computed in SDC stage now)
        # A.2 (o) has inputs from b1.out and b2.out
        assert "b1.out" in ir.operations["A.2"].inputs
        assert "b2.out" in ir.operations["A.2"].inputs

        # Check sched_vars
        assert ir.sched_vars == {"t_b1", "t_b2", "t_b3"}

        # II is unknown (scheduler decides)
        assert ir.ii is None

    def test_anonymous_sched_var(self):
        """Anonymous scheduler variable (?)."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: ?>() -> () {
                A := new Add[32];
                x := A<?>( a, b);
            }
        ''')

        op = ir.operations["A.0"]
        assert op.timing.sched_var is None
        assert op.timing.is_anonymous()

    def test_event_based_timing(self):
        """Event-based timing ('G+1)."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<'G>(a, b);
            }
        ''')

        op = ir.operations["A.0"]
        assert op.timing.base_event == "G"
        assert op.timing.sched_var is None

    # ========================================================================
    # Pool Tests
    # ========================================================================

    def test_pool_resource(self):
        """Pool declared as resource."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                pool Adders : Add[32] * 2;
                x := Adders[?]<?t>(a, b);
            }
        ''')

        assert "Adders" in ir.pools
        pool = ir.pools["Adders"]
        assert pool.is_pool is True
        assert pool.max_instances == 2
        assert pool.module == "Add"

    def test_pool_unbounded(self):
        """Unbounded pool."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                pool A : Add[32] * ?;
            }
        ''')

        assert ir.pools["A"].max_instances is None
        assert ir.pools["A"].is_pool is True

    def test_pool_invocation_with_binding(self):
        """Pool invocation captures binding."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                pool A : Add[32] * 2;
                x := A[?idx]<?t>(a, b);
            }
        ''')

        op = ir.operations["A.0"]
        assert op.binding is not None
        assert op.binding.kind == "named"
        assert op.binding.value == "idx"

    # ========================================================================
    # Timing Constraint Tests
    # ========================================================================

    def test_timing_with_offset(self):
        """Timing with fixed offset (?t+1)."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t+1>(a, b);
            }
        ''')

        op = ir.operations["A.0"]
        assert op.timing.sched_var == "t"
        assert op.timing.offset == 1

    def test_timing_with_range(self):
        """Timing with range offset ('G+[4..])."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<'G+[4..6]>(a, b);
            }
        ''')

        op = ir.operations["A.0"]
        assert op.timing.base_event == "G"
        assert op.timing.range_lo == 4
        assert op.timing.range_hi == 6

    def test_timing_range_lo_only(self):
        """Timing with lower-bounded range ('G+[4..])."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<'G+[4..]>(a, b);
            }
        ''')

        op = ir.operations["A.0"]
        assert op.timing.range_lo == 4
        assert op.timing.range_hi is None

    # ========================================================================
    # Custom Latency Tests
    # ========================================================================

    def test_custom_latency(self):
        """Custom module latencies stored in IR."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                M := new Mult[32];
                x := M<?t1>(a, b);
                y := M<?t2>(x.out, c);
            }
        ''', module_latencies={"Mult": 3})

        # Check latency stored for instance
        assert ir.module_latencies["M"] == 3

    def test_ii_fixed(self):
        """Fixed II (initiation interval)."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 2>() -> () {}
        ''')
        assert ir.ii == 2

    def test_ii_scheduler_decides(self):
        """II decided by scheduler."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: ?>() -> () {}
        ''')
        assert ir.ii is None

    # ========================================================================
    # Interface (Port) Tests
    # ========================================================================

    def test_input_port_timing(self):
        """Input ports with event-based timing."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>(
                a: ['G, 'G+1] 32,
                b: ['G+2, 'G+3] 16
            ) -> () {}
        ''')

        # Check input ports
        assert "a" in ir.inputs
        assert "b" in ir.inputs

        # Port a: ['G, 'G+1] 32
        a = ir.inputs["a"]
        assert a.width == 32
        assert a.start_event == "G"
        assert a.start_offset is None
        assert a.end_event == "G"
        assert a.end_offset == 1

        # Port b: ['G+2, 'G+3] 16
        b = ir.inputs["b"]
        assert b.width == 16
        assert b.start_event == "G"
        assert b.start_offset == 2
        assert b.end_event == "G"
        assert b.end_offset == 3

    def test_output_port_timing(self):
        """Output ports with anonymous timing."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: ?>() -> (
                out: [?, ?] 32
            ) {}
        ''')

        assert "out" in ir.outputs
        out = ir.outputs["out"]
        assert out.width == 32
        # Anonymous timing (scheduler decides)
        assert out.start_sched_var == ""  # empty string for anonymous ?
        assert out.end_sched_var == ""

    # ========================================================================
    # SpaceLoop Tests
    # ========================================================================

    def test_spaceloop_unroll_instances(self):
        """SpaceLoop unrolls instances with _i suffix."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>(
                x: ['G, 'G+1] 32,
                y: ['G, 'G+1] 32,
            ) -> () where W > 0 {
                spaceloop<'G> for i in 0..2 {
                    M := new Mult[32];
                    r := M<'G>(x, y);
                }
            }
        ''', module_latencies={"Mult": 3})

        # Two unrolled instances
        assert "M_0" in ir.module_latencies
        assert "M_1" in ir.module_latencies
        assert ir.module_latencies["M_0"] == 3
        assert ir.module_latencies["M_1"] == 3

        # Two unrolled operations
        assert "M_0.0" in ir.operations
        assert "M_1.0" in ir.operations

        # Result wires suffixed
        assert ir.result_to_op["r_0"] == "M_0.0"
        assert ir.result_to_op["r_1"] == "M_1.0"

        # No loops in the IR (spaceloop is flattened)
        assert ir.loops == []

    def test_spaceloop_trip_count_4(self):
        """SpaceLoop with trip count 4 produces 4 copies."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>(
                x: ['G, 'G+1] 32,
                y: ['G, 'G+1] 32,
            ) -> () {
                spaceloop<'G> for i in 0..4 {
                    A := new Add[32];
                    r := A<'G>(x, y);
                }
            }
        ''')

        for i in range(4):
            assert f"A_{i}" in ir.module_latencies
            assert f"A_{i}.0" in ir.operations
            assert ir.result_to_op[f"r_{i}"] == f"A_{i}.0"

    def test_spaceloop_pool(self):
        """SpaceLoop with pool in body: pool keeps original name (shared resource)."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>(x: ['G, 'G+1] 32, y: ['G, 'G+1] 32,) -> () {
                spaceloop<'G> for i in 0..2 {
                    pool P : Add[32] * 2;
                    r := P[?]<'G>(x, y);
                }
            }
        ''')

        # Pool stays as "P" — shared resource, not suffixed per iteration
        assert "P" in ir.pools
        assert "P_0" not in ir.pools
        assert "P_1" not in ir.pools
        # Both unrolled ops reference the same pool P
        assert ir.result_to_op["r_0"] == "P.0"
        assert ir.result_to_op["r_1"] == "P.1"

    # ========================================================================
    # TimeLoop Tests
    # ========================================================================

    def test_timeloop_region_created(self):
        """TimeLoop creates a LoopRegion in ir.loops."""
        ir = self.parse_and_build_ir('''
            comp Foo[N, W]<'G: ?>(
                a: ['G, 'G+1] W,
                b: ['G, 'G+1] W,
            ) -> () where N > 0, W > 0 {
                pool M : Mult[W] * 2;
                timeloop<'G> [II=?] for i in 0..4 {
                    m := M[?]<?t>(a, b);
                }
            }
        ''', module_latencies={"Mult": 3})

        assert len(ir.loops) == 1
        region = ir.loops[0]
        assert isinstance(region, LoopRegion)
        assert region.loop_var == "i"
        assert region.trip_count == 4
        assert region.ii is None  # [II=?] -> solver decides

    def test_timeloop_fixed_ii(self):
        """TimeLoop with fixed II stores the value."""
        ir = self.parse_and_build_ir('''
            comp Foo[N]<'G: 1>(x: ['G, 'G+1] 32, y: ['G, 'G+1] 32,) -> () {
                pool M : Mult[32] * 1;
                timeloop<'G> [II=4] for i in 0..8 {
                    m := M[?]<?t>(x, y);
                }
            }
        ''', module_latencies={"Mult": 3})

        region = ir.loops[0]
        assert region.ii == 4
        assert region.trip_count == 8

    def test_timeloop_body_ops(self):
        """TimeLoop body operations are captured in the region."""
        ir = self.parse_and_build_ir('''
            comp Foo[N]<'G: ?>(
                a: ['G, 'G+1] 32,
                b: ['G, 'G+1] 32,
            ) -> () {
                pool M : Mult[32] * 2;
                pool A : Add[32] * 1;
                timeloop<'G> [II=?] for i in 0..4 {
                    m := M[?u]<?t1>(a, b);
                    r := A[?v]<?t2>(m.out, a);
                }
            }
        ''', module_latencies={"Mult": 3, "Add": 1})

        region = ir.loops[0]
        assert "M.0" in region.body_ops
        assert "A.0" in region.body_ops

        m_op = region.body_ops["M.0"]
        assert m_op.result == "m"
        assert "a" in m_op.inputs
        assert "b" in m_op.inputs

        r_op = region.body_ops["A.0"]
        assert r_op.result == "r"
        assert "m.out" in r_op.inputs

        assert region.body_result_to_op["m"] == "M.0"
        assert region.body_result_to_op["r"] == "A.0"

    def test_timeloop_body_instances(self):
        """TimeLoop body `new` instances go into body_instances."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: ?>(x: ['G, 'G+1] 32, y: ['G, 'G+1] 32,) -> () {
                timeloop<'G> [II=?] for i in 0..3 {
                    M := new Mult[32];
                    m := M<?t>(x, y);
                }
            }
        ''', module_latencies={"Mult": 3})

        region = ir.loops[0]
        assert "M" in region.body_instances
        assert region.body_instances["M"] == 3
        # Not unrolled into flat IR
        assert "M" not in ir.module_latencies

    def test_timeloop_not_unrolled_into_flat(self):
        """TimeLoop operations stay in the region, not the flat IR."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: ?>(x: ['G, 'G+1] 32, y: ['G, 'G+1] 32,) -> () {
                pool M : Mult[32] * 1;
                timeloop<'G> [II=?] for i in 0..4 {
                    m := M[?]<?t>(x, y);
                }
            }
        ''', module_latencies={"Mult": 3})

        # No operations in flat IR from the loop body
        assert ir.operations == {}
        # Region has the op
        assert "M.0" in ir.loops[0].body_ops

    def test_mixed_flat_and_timeloop(self):
        """Flat ops and a timeloop can coexist in the same component."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>(x: ['G, 'G+1] 32, y: ['G, 'G+1] 32,) -> () {
                A := new Add[32];
                pre := A<'G>(x, y);
                pool M : Mult[32] * 1;
                timeloop<'G> [II=2] for i in 0..4 {
                    m := M[?]<?t>(x, y);
                }
            }
        ''', module_latencies={"Mult": 3, "Add": 1})

        # Flat op present
        assert "A.0" in ir.operations
        assert ir.result_to_op["pre"] == "A.0"

        # Loop region separate
        assert len(ir.loops) == 1
        assert "M.0" in ir.loops[0].body_ops

    def test_mixed_port_timing(self):
        """Mix of fixed input and unknown output timing."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: ?>(
                a: ['G, 'G+1] 32
            ) -> (
                out: [?, ?] 32
            ) {}
        ''')

        # Input has fixed timing
        assert ir.inputs["a"].start_event == "G"
        assert ir.inputs["a"].is_fixed()

        # Output has unknown timing
        assert ir.outputs["out"].start_sched_var == ""
        assert not ir.outputs["out"].is_fixed()

    # ========================================================================
    # Bundle Tests
    # ========================================================================

    def test_bundle_port_in_ir(self):
        """Bundle port in signature propagates bundle_size to PortTiming."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>(
                a[4]: ['G, 'G+1] 32,
            ) -> () {}
        ''')
        assert "a" in ir.inputs
        pt = ir.inputs["a"]
        assert pt.bundle_size == 4
        assert pt.width == 32
        assert pt.start_event == "G"

    def test_bundle_decl_in_ir(self):
        """Standalone bundle decl appears in ir.bundles."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                bundle twiddle[2]: ['G, 'G+1] 32;
            }
        ''')
        assert "twiddle" in ir.bundles
        info = ir.bundles["twiddle"]
        assert isinstance(info, BundleInfo)
        assert info.bundle_size == 2
        assert info.width == 32
        assert info.start_event == "G"

    def test_bundle_decl_in_timeloop(self):
        """Bundle decl inside timeloop goes to region.body_bundles."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: ?>(a: ['G, 'G+1] 32,) -> () {
                pool M : Mult[32] * 1;
                timeloop<'G> [II=?] for i in 0..4 {
                    bundle partial[2]: ['G, 'G+1] 32;
                    m := M[?]<?t>(a, a);
                }
            }
        ''', module_latencies={"Mult": 3})
        region = ir.loops[0]
        assert "partial" in region.body_bundles
        info = region.body_bundles["partial"]
        assert info.bundle_size == 2

    def test_bundle_access_as_invocation_arg(self):
        """Bundle element access encoded as 'name{index}' in op inputs."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                pool A : Add[32] * 1;
                r := A[?]<?t>(x{0}, y{1});
            }
        ''')
        op = ir.operations["A.0"]
        assert "x{0}" in op.inputs
        assert "y{1}" in op.inputs

    def test_qualified_bundle_access_as_arg(self):
        """inst.buf{i} encodes as 'inst.buf{i}' in op inputs."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                pool A : Add[32] * 1;
                r := A[?]<?t>(src.data{2}, b);
            }
        ''')
        op = ir.operations["A.0"]
        assert "src.data{2}" in op.inputs

    def test_scalar_port_bundle_size_none_in_ir(self):
        """Scalar port PortTiming has bundle_size=None."""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>(x: ['G, 'G+1] 32,) -> () {}
        ''')
        assert ir.inputs["x"].bundle_size is None

    # ========================================================================
    # Named Time Bind Tests
    # ========================================================================

    def test_named_bind_alias_in_flat_op(self):
        """?t='G+1 sets tc.alias='t', tc.base_event='G', tc.offset=1; alias NOT in sched_vars"""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t='G+1>(a, b);
            }
        ''')
        op = ir.operations["A.0"]
        assert op.timing.base_event == "G"
        assert op.timing.offset == 1
        assert op.timing.alias == "t"
        # Aliases are NOT free variables — they name existing timing vars
        assert "t" not in ir.sched_vars

    def test_named_bind_alias_in_timeloop_start(self):
        """timeloop<?tloop='G> sets region.start.alias='tloop'; alias NOT in sched_vars"""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                pool M : Mult[32] * 1;
                timeloop<?tloop='G> [II=?] for i in 0..4 {
                    m := M[?]<?t>(a, b);
                }
            }
        ''', module_latencies={"Mult": 3})
        region = ir.loops[0]
        assert region.start.alias == "tloop"
        # Aliases are NOT free variables — they name the loop-start event anchor
        assert "tloop" not in ir.sched_vars

    def test_named_bind_alias_in_loop_body_op(self):
        """Alias on invocation time inside timeloop: alias NOT in region.sched_vars"""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                pool M : Mult[32] * 1;
                timeloop<'G> [II=?] for i in 0..4 {
                    m := M[?]<?alias='G>(a, b);
                }
            }
        ''', module_latencies={"Mult": 3})
        region = ir.loops[0]
        op = region.body_ops["M.0"]
        assert op.timing.alias == "alias"
        # Aliases are NOT free variables — they name existing timing vars
        assert "alias" not in region.sched_vars

    def test_plain_time_has_alias_none_in_ir(self):
        """Plain ?t timing has alias=None in TC"""
        ir = self.parse_and_build_ir('''
            comp Foo<'G: 1>() -> () {
                A := new Add[32];
                x := A<?t>(a, b);
            }
        ''')
        assert ir.operations["A.0"].timing.alias is None


# ============================================================================
# Run tests directly
# ============================================================================

def run_tests():
    """Run all tests and report results."""
    import traceback

    test = TestIR()
    methods = [m for m in dir(test) if m.startswith("test_")]

    passed = 0
    failed = 0

    for method in methods:
        test.setup_method()
        try:
            getattr(test, method)()
            print(f"  ✓ {method}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {method}")
            print(f"    {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("Running IR tests...\n")
    success = run_tests()
    exit(0 if success else 1)
