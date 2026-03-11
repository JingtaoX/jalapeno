# Jalapeno Compiler — Architecture & Design Notes

## Project Overview

Jalapeno is an HLS scheduling language (Filament-style) with `?` for solver-determined
values. The compiler takes `.fil` source files, builds an IR, generates scheduling
constraints (SDC-style), solves them with Z3, and (eventually) emits Verilog.

```
Run tests:    venv/bin/python tests/test_grammar.py
Run compiler: venv/bin/python src/main.py <file> [--show ast|ir|sdc|schedule|emit] [--stop-after STAGE]
```

---

## Package Structure

```
src/
  ast_nodes.py          — all AST dataclasses
  ast_builder.py        — Lark Transformer -> AST
  parser.py             — creates Lark parser
  ir/
    nodes.py            — IR dataclasses: TimingConstraint, Operation, Resource,
                          LoopRegion, SchedulingIR, PortTiming
    builder.py          — IRBuilder, build_ir()
    __init__.py         — re-exports everything (from ir import ... still works)
  ir.py                 — thin shim: from ir.nodes import *; from ir.builder import *
  sdc/
    model.py            — ConstraintType, Constraint, SDCModel
    generator.py        — SDCGenerator, generate_sdc()
    __init__.py         — re-exports everything
  sdc.py                — thin shim: from sdc.model import *; from sdc.generator import *
  solver.py             — Z3-based solver, Schedule, SolveStatus
  codegen.py            — Codegen, generate_code()
  main.py               — pipeline + CLI
grammar/
  jalapeno.lark
tests/
  test_grammar.py       (73 tests)
  test_ir.py            (36 tests)
  test_sdc.py           (20 tests)
examples/
  dot_product.fil, three_add.fil, pool_binding.fil
  timeloop.fil, two_mul_time.fil, two_mul_space.fil
  nested_loop.fil       — 101 demo: timeloop outer + spaceloop inner
benchmark/
  gemm/                 — gemm.c, gemm.h
  jacobi-1d/            — jacobi-1d.c, jacobi-1d.h
  mvt/                  — mvt.c, mvt.h
```

---

## Key Design Decisions

### Loop IR Lowering

- **SpaceLoop** → unrolled into flat `ir.operations` with `_i` name suffix
- **TimeLoop** → `LoopRegion` in `ir.loops` (symbolic, not unrolled)
- **Nested** (spaceloop inside timeloop) → spaceloop unrolled *into* the `LoopRegion` body

### LoopRegion Fields

```
loop_var, start (TimingConstraint), ii (int|None), trip_count (int)
body_instances {name->latency}, body_pools {name->Resource}
body_ops {op_name->Operation}, body_result_to_op {wire->op_name}
sched_vars {str}
```

### Grammar Conventions

- `timeloop<'G> [II=4] for i in 0..N { ... }` — start time required
- `spaceloop<'G> for i in 0..N { ... }` — start time required
- `pool A : Add[32] * 2;` — pool size is upper bound, no `..` prefix
- `--show ast ir sdc` and `--stop-after ir` are orthogonal

### Bundle Feature (fully implemented)

- Grammar: `port_bundle_signal` (name[N]: interval expr), `bundle_decl` command,
  `port_bundle_simple`/`port_bundle_qualified` in port_expr
- AST: `PortDef.bundle_size`, `PortBundleAccess`, `BundleDecl`
- IR: `PortTiming.bundle_size`, `BundleInfo`, `SchedulingIR.bundles`, `LoopRegion.body_bundles`
- IR display: shows `name[N]: interval width` for bundle ports
- Bundle element access: `a{i}` → `PortBundleAccess`, encoded as `"name{index}"` in IR op inputs

### Reg Component

- Used as accumulator primitive: `pool R : Reg[32] * 1;`
- Latency comes from external config (same as Mult/Add); no special AST treatment

### Named Time Bind Feature (alias)

- Syntax: `?name=time` anywhere a `time` appears — names an existing timing var; NOT a new free var
- Examples: `timeloop<?tloop='G>`, `m := M[?]<?t='G+1>(a, b);`
- AST: `Time.alias: Optional[str]` — set by `time_named_bind` transformer
- IR: `TimingConstraint.alias: Optional[str]` — propagated in `_extract_timing`
- **Aliases are NOT added to sched_vars** — they name existing op.t or event anchors
- `ir/builder.py` post-processing removes aliases from `ir.sched_vars` and `region.sched_vars`
  after `_process_timeloop` body is built
- `_alias_map: Dict[str, Optional[str]]` in SDC generator: alias_name → op_timing_var or None

### TimeLoop / Modulo Scheduling (fully implemented)

- `LoopRegion` captures body_ops, body_pools, body_bundles, body_connects, start TimingConstraint
- SDC generates: (a) LOOP_INTRA_DEP, (b) LOOP_RECURRENCE, (c) LOOP_MRT, (d) LOOP_II_NON_NEG
- Recurrence: from `body_connects` wire chain `s{i+1} = producer.out` → `II >= ceil(lat/1)`
- MRT: `II >= ceil(k*lat/N)` per resource pool
- Solver minimizes `?II_N` as secondary objective after makespan

### Resource / Non-overlap Constraint Generation

- RESOURCE: always emitted when `len(ops) > 0` (not just `> max_instances`) so solver assigns bindings
- NON_OVERLAP: only emitted when `len(ops) > max_instances` (no conflict when each op has its own unit)
- This ensures parallel same-time ops in different pool units are not over-constrained

---

## Test Counts (all passing)

73 grammar + 36 IR + 20 SDC = 129 total
