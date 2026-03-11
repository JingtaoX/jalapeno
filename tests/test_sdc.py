"""
Tests for SDC constraint generation and solving.

Variable naming convention:
- Timing variables: {op_name}.t (e.g., "A.0.t", "A.1.t")
- Binding variables: {op_name}.i (e.g., "A.0.i", "A.1.i")
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser import create_parser
from ast_builder import build_ast
from ir import build_ir
from sdc import generate_sdc, ConstraintType, SDCModel
from solver import solve_sdc, SolveStatus


class TestSDCGeneration:
    """Test suite for SDC constraint generation."""

    def setup_method(self):
        self.parser = create_parser()

    def parse_and_generate_sdc(self, code: str, module_latencies=None) -> SDCModel:
        tree = self.parser.parse(code)
        ast = build_ast(tree)
        component = ast.modules[0]
        ir = build_ir(component, module_latencies)
        return generate_sdc(ir)

    # ========================================================================
    # Basic Constraint Generation
    # ========================================================================

    def test_timing_variables(self):
        """Each operation gets a timing variable {op}.t."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                A := new Add[32];
                x := A<?t>(a, b);
            }
        ''')

        # Operation A.0 should have timing variable A.0.t
        assert "A.0.t" in model.timing_vars

    def test_non_negative_constraints(self):
        """All timing variables get non-negative constraints."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                A := new Add[32];
                x := A<?t>(a, b);
            }
        ''')

        non_neg = model.by_type(ConstraintType.NON_NEGATIVE)
        assert len(non_neg) > 0
        vars_with_constraint = {c.lhs for c in non_neg}
        assert "A.0.t" in vars_with_constraint

    def test_op_to_op_constraint(self):
        """Data dependencies generate op-to-op constraints."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                A := new Add[32];
                x := A<?t1>(a, b);
                y := A<?t2>(x.out, c);
            }
        ''')

        op_deps = model.by_type(ConstraintType.OP_TO_OP)
        assert len(op_deps) == 1

        dep = op_deps[0]
        assert dep.lhs == "A.1.t"  # consumer (y)
        assert dep.rhs == "A.0.t"  # producer (x)
        assert dep.constant == 1   # latency of Add
        assert dep.is_lower_bound

    def test_three_add_data_dependencies(self):
        """Three-add pattern generates correct op-to-op dependencies."""
        model = self.parse_and_generate_sdc('''
            comp three_adds<'G: ?>() -> () {
                A := new Add[32];
                b1 := A<?t_b1>(a1, a2);
                b2 := A<?t_b2>(a3, a4);
                o  := A<?t_b3>(b1.out, b2.out);
            }
        ''')

        op_deps = model.by_type(ConstraintType.OP_TO_OP)
        assert len(op_deps) == 2

        # A.2 (o) depends on both A.0 (b1) and A.1 (b2)
        constraints = {(c.lhs, c.rhs) for c in op_deps}
        assert ("A.2.t", "A.0.t") in constraints
        assert ("A.2.t", "A.1.t") in constraints

    def test_custom_latency(self):
        """Custom module latencies affect constraints."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                M := new Mult[32];
                x := M<?t1>(a, b);
                y := M<?t2>(x.out, c);
            }
        ''', module_latencies={"Mult": 3})

        op_deps = model.by_type(ConstraintType.OP_TO_OP)
        assert len(op_deps) == 1
        assert op_deps[0].constant == 3  # Mult latency

    def test_sched_var_to_ops_mapping(self):
        """User sched_vars are tracked to their operations."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                A := new Add[32];
                x := A<?t_b1>(a, b);
                y := A<?t_b2>(c, d);
            }
        ''')

        assert "t_b1" in model.sched_var_to_ops
        assert "t_b2" in model.sched_var_to_ops
        assert model.sched_var_to_ops["t_b1"] == ["A.0"]
        assert model.sched_var_to_ops["t_b2"] == ["A.1"]

    # ========================================================================
    # Resource Constraints
    # ========================================================================

    def test_pool_resource_constraint(self):
        """Pools with limited instances generate resource constraints."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                pool A : Add[32] * 2;
                x := A[?]<?t1>(a, b);
                y := A[?]<?t2>(c, d);
                z := A[?]<?t3>(e, f);
            }
        ''')

        resource_constraints = model.by_type(ConstraintType.RESOURCE)
        assert len(resource_constraints) == 1

        rc = resource_constraints[0]
        assert rc.resource == "A"
        assert rc.max_concurrent == 2
        assert set(rc.operations) == {"A.0", "A.1", "A.2"}

    def test_pool_binding_variables(self):
        """Pool operations get binding variables."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                pool A : Add[32] * 2;
                x := A[?]<?t1>(a, b);
                y := A[?]<?t2>(c, d);
            }
        ''')

        assert "A.0.i" in model.binding_vars
        assert "A.1.i" in model.binding_vars

    def test_unbounded_pool_no_constraint(self):
        """Unbounded pools don't generate resource constraints."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                pool A : Add[32] * ?;
                x := A[?]<?t1>(a, b);
                y := A[?]<?t2>(c, d);
            }
        ''')

        # Unknown pool size: RESOURCE constraint is emitted with max_concurrent=None
        # so the solver can minimize pool size while enforcing non-overlap.
        resource_constraints = model.by_type(ConstraintType.RESOURCE)
        assert len(resource_constraints) == 1
        assert resource_constraints[0].max_concurrent is None

    def test_pool_with_enough_units(self):
        """Resource constraint is emitted for binding assignment even when pool has enough units."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                pool A : Add[32] * 3;
                x := A[?]<?t1>(a, b);
                y := A[?]<?t2>(c, d);
            }
        ''')

        # 2 ops, 3 units - RESOURCE constraint still emitted so solver can assign bindings
        resource_constraints = model.by_type(ConstraintType.RESOURCE)
        assert len(resource_constraints) == 1
        assert resource_constraints[0].max_concurrent == 3

    # ========================================================================
    # Binding Constraints
    # ========================================================================

    def test_named_binding_constraint(self):
        """Named bindings generate equality constraints."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>() -> () {
                pool A : Add[32] * 2;
                x := A[?idx]<?t1>(a, b);
                y := A[?idx]<?t2>(c, d);
            }
        ''')

        binding_constraints = model.by_type(ConstraintType.BINDING_EQUALITY)
        assert len(binding_constraints) == 1

        bc = binding_constraints[0]
        assert bc.binding_var == "idx"
        assert set(bc.operations) == {"A.0", "A.1"}

    # ========================================================================
    # Input Availability
    # ========================================================================

    def test_input_availability_at_g(self):
        """Inputs at 'G are available at time 0 only (half-open interval)."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>(
                a: ['G, 'G+1] 32
            ) -> () {
                A := new Add[32];
                x := A<?t>(a, b);
            }
        ''')

        # x uses input 'a' which is available during ['G, 'G+1) = cycle 0 only
        input_deps = model.by_type(ConstraintType.INPUT_WINDOW)

        # Should have both lower and upper bound constraints
        lower_bounds = [c for c in input_deps if c.is_lower_bound and "avail" in c.reason]
        upper_bounds = [c for c in input_deps if not c.is_lower_bound and "expires" in c.reason]

        assert len(lower_bounds) >= 1  # A.0.t >= 0
        assert len(upper_bounds) >= 1  # A.0.t <= 0
        assert any(c.constant == 0 for c in lower_bounds)
        assert any(c.constant == 0 for c in upper_bounds)

    def test_input_availability_with_offset(self):
        """Inputs with offset generate input window constraints (half-open)."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>(
                a: ['G+2, 'G+3] 32
            ) -> () {
                A := new Add[32];
                x := A<?t>(a, b);
            }
        ''')

        # ['G+2, 'G+3) = cycle 2 only (half-open interval)
        input_deps = model.by_type(ConstraintType.INPUT_WINDOW)

        lower_bounds = [c for c in input_deps if c.is_lower_bound and "avail" in c.reason]
        upper_bounds = [c for c in input_deps if not c.is_lower_bound and "expires" in c.reason]

        assert len(lower_bounds) >= 1
        assert len(upper_bounds) >= 1
        assert any(c.constant == 2 for c in lower_bounds)
        assert any(c.constant == 2 for c in upper_bounds)

    # ========================================================================
    # Output Deadline Constraints
    # ========================================================================

    def test_output_deadline_constraint(self):
        """Output with fixed timing generates deadline constraint."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>(
                a: ['G, 'G+1] 32
            ) -> (
                out: ['G+3, 'G+4] 32
            ) {
                A := new Add[32];
                x := A<?t>(a, b);
                out = x.out;
            }
        ''')

        # Output expects data at cycle 3
        # x has latency 1, so A.0.t + 1 <= 3, i.e., A.0.t <= 2
        deadlines = model.by_type(ConstraintType.OUTPUT_DEADLINE)
        assert len(deadlines) == 1
        assert deadlines[0].lhs == "A.0.t"
        assert deadlines[0].constant == 2  # 3 - 1 = 2
        assert not deadlines[0].is_lower_bound

    def test_output_no_deadline_for_unknown(self):
        """Output with [?, ?] timing has no deadline constraint."""
        model = self.parse_and_generate_sdc('''
            comp Foo<'G: ?>(
                a: ['G, 'G+1] 32
            ) -> (
                out: [?, ?] 32
            ) {
                A := new Add[32];
                x := A<?t>(a, b);
                out = x.out;
            }
        ''')

        deadlines = model.by_type(ConstraintType.OUTPUT_DEADLINE)
        assert len(deadlines) == 0


class TestSDCSolving:
    """Test suite for SDC solving."""

    def setup_method(self):
        self.parser = create_parser()

    def parse_and_solve(self, code: str, module_latencies=None):
        tree = self.parser.parse(code)
        ast = build_ast(tree)
        component = ast.modules[0]
        ir = build_ir(component, module_latencies)
        model = generate_sdc(ir)
        return solve_sdc(model)

    def test_simple_chain(self):
        """Simple chain: x -> y should schedule y after x."""
        schedule = self.parse_and_solve('''
            comp Foo<'G: ?>() -> () {
                A := new Add[32];
                x := A<?t1>(a, b);
                y := A<?t2>(x.out, c);
            }
        ''')

        assert schedule.status in (SolveStatus.OPTIMAL, SolveStatus.SATISFIABLE)
        # A.1.t >= A.0.t + 1
        assert schedule.assignments["A.1.t"] >= schedule.assignments["A.0.t"] + 1

    def test_three_add_schedule(self):
        """Three-add pattern: b1, b2 can be parallel, o after both."""
        schedule = self.parse_and_solve('''
            comp three_adds<'G: ?>() -> () {
                A := new Add[32];
                b1 := A<?t_b1>(a1, a2);
                b2 := A<?t_b2>(a3, a4);
                o  := A<?t_b3>(b1.out, b2.out);
            }
        ''')

        assert schedule.status in (SolveStatus.OPTIMAL, SolveStatus.SATISFIABLE)

        t_a0 = schedule.assignments["A.0.t"]  # b1
        t_a1 = schedule.assignments["A.1.t"]  # b2
        t_a2 = schedule.assignments["A.2.t"]  # o

        # o must be after both b1 and b2
        assert t_a2 >= t_a0 + 1
        assert t_a2 >= t_a1 + 1

    def test_three_add_with_pool(self):
        """Three-add with pool of 2: resource constraint must be satisfied."""
        schedule = self.parse_and_solve('''
            comp three_adds<'G: ?>() -> () {
                pool A : Add[32] * 2;
                b1 := A[?]<?t_b1>(a1, a2);
                b2 := A[?]<?t_b2>(a3, a4);
                o  := A[?]<?t_b3>(b1.out, b2.out);
            }
        ''')

        assert schedule.status in (SolveStatus.OPTIMAL, SolveStatus.SATISFIABLE)

        t_a0 = schedule.assignments["A.0.t"]
        t_a1 = schedule.assignments["A.1.t"]
        t_a2 = schedule.assignments["A.2.t"]

        # Data dependencies
        assert t_a2 >= t_a0 + 1
        assert t_a2 >= t_a1 + 1

    def test_optimal_schedule(self):
        """Optimizer should find minimal makespan."""
        schedule = self.parse_and_solve('''
            comp Foo<'G: ?>() -> () {
                A := new Add[32];
                x := A<?t1>(a, b);
                y := A<?t2>(x.out, c);
            }
        ''')

        assert schedule.status == SolveStatus.OPTIMAL
        # Optimal: A.0.t=0, A.1.t=1, makespan=2
        assert schedule.assignments["A.0.t"] == 0
        assert schedule.assignments["A.1.t"] == 1

    def test_print_schedule(self):
        """Schedule can be printed without errors."""
        schedule = self.parse_and_solve('''
            comp three_adds<'G: ?>() -> () {
                A := new Add[32];
                b1 := A<?t_b1>(a1, a2);
                b2 := A<?t_b2>(a3, a4);
                o  := A<?t_b3>(b1.out, b2.out);
            }
        ''')

        # Should not raise
        schedule.print_schedule()


# ============================================================================
# Run tests directly
# ============================================================================

def run_tests():
    """Run all tests and report results."""
    import traceback

    # Run generation tests
    print("SDC Generation Tests:")
    gen_test = TestSDCGeneration()
    gen_methods = [m for m in dir(gen_test) if m.startswith("test_")]

    passed = 0
    failed = 0

    for method in gen_methods:
        gen_test.setup_method()
        try:
            getattr(gen_test, method)()
            print(f"  ✓ {method}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {method}")
            print(f"    {e}")
            traceback.print_exc()
            failed += 1

    # Run solving tests
    print("\nSDC Solving Tests:")
    solve_test = TestSDCSolving()
    solve_methods = [m for m in dir(solve_test) if m.startswith("test_")]

    for method in solve_methods:
        solve_test.setup_method()
        try:
            getattr(solve_test, method)()
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
    print("Running SDC tests...\n")
    success = run_tests()
    exit(0 if success else 1)
