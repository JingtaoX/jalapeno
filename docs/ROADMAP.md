# Jalapeno — End-to-End Pipeline Roadmap

> Not tracked by git (`.claude/` is in `.gitignore`).
> Last updated: 2026-03-11

---

## Goal

Run the three PolyBench/C starter kernels end-to-end through the Jalapeno compiler,
synthesize the generated Verilog with Vivado, and deploy on two different FPGAs with
different resource constraints — demonstrating feasibility-guided HLS scheduling.

Kernels:
- `benchmark/gemm/` — C = alpha * A * B + beta * C  (triple loop, MAC-heavy)
- `benchmark/mvt/` — Matrix-Vector product Transpose (two double loops, accumulate)
- `benchmark/jacobi-1d/` — iterative 1D stencil (time-stepped, dependency across iters)

---

## Roadmap Items

### (1) Verilog Code Emit  [HIGHEST PRIORITY]

**What:** `--show emit` / `--stop-after emit` produces synthesizable Verilog RTL from
the solved schedule.

**Key questions to settle first:**

| Question | Current thinking |
|---|---|
| Abstraction level | We are at the **operator scheduling** level, above RTL registers. Values are wires with compile-time latencies (like Filament). Registers are *implicit* — inserted wherever a value must be held across a clock cycle boundary. |
| Do we add explicit `Reg` to the language? | **No for the user-facing language.** The codegen inserts pipeline registers automatically where `(use_time - produce_time) > 0`. The `Reg` pool already exists for *accumulator feedback* (RAW dependency) which is semantically different — keep that. |
| Memory interface | Start with simple port-level I/O (arrays passed as flat AXI-stream or ready/valid ports). Memory subsystem is out of scope for now. |
| Clock/reset | Standard synchronous, single-clock, active-high synchronous reset. |

**Sub-tasks:**
1. Define `CodegenContext`: maps each `(op_name, output_wire)` → `(cycle_offset, signal_name)`.
2. For each scheduled operation, emit: `always_ff` block or `assign` depending on whether
   it's a registered output.
3. Auto-insert pipeline registers: when `use_cycle - produce_cycle > 0`, chain through
   `reg [W-1:0] pipe_NAME_N` signals.
4. Emit module header (ports), body (instances + pipe regs), and footer.
5. Handle `timeloop` bodies: emit as always_ff loop with II-based enable logic.
6. Handle `spaceloop` (already unrolled in IR): emit as replicated combinational/FF chains.

**Open design question:** Should we target **structural Verilog** (explicit instances of
primitive modules like `add_32`, `mul_32`) or **behavioral Verilog** (`+`, `*` operators
and let Vivado infer)? Behavioral is easier to emit and Vivado handles it well. Recommend
behavioral for now with a flag to switch later.

---

### (2) User-Level Language (C-like with annotations)  [MEDIUM PRIORITY]

**Motivation:** The current Filament-style language is expressive but verbose. For the
benchmarks we need to write real programs without manually computing every timing offset.

**Proposed approach — "Jalapeno-C":**
- Looks like C with `for` loops, assignments, scalar and array variables.
- Annotations on loops and ops that partially constrain scheduling:
  - `#[timeloop(II=?)]` on a for-loop → becomes `timeloop` with solver-determined II.
  - `#[spaceloop]` → fully unroll.
  - `#[bind(pool=Mult, instances=2)]` → resource binding hint.
  - `#[latency(3)]` on a function call → override latency assumption.
- Compilation: Jalapeno-C → Jalapeno `.fil` → existing pipeline.

**Why this matters for benchmarks:** Writing `gemm` by hand in the current `.fil` syntax
is painful (explicit timing on every operation). Jalapeno-C would let us write it close
to the polybench C source and just add a few annotations.

**Rough priority call:** Defer until (1) and (3) are working. Once we can generate Verilog
for hand-written `.fil` programs, revisit whether Jalapeno-C is needed to write the bigger
benchmarks.

---

### (3) Full Feasibility-Checking Pipeline  [HIGH PRIORITY, parallel with (1)]

**What:** Given a `.fil` program and a target FPGA resource profile, check whether a valid
schedule exists *before* attempting synthesis.

**FPGA resource profiles (two targets):**
- FPGA-A: (fill in — e.g., Artix-7 or Zynq-7000, smaller)
- FPGA-B: (fill in — e.g., Zynq UltraScale+, larger)

**Pipeline stages:**
```
.fil file
  → Parser + IR builder           (existing)
  → SDC constraint generation      (existing)
  → Z3 solver                      (existing)
       ↓ UNSAT → "infeasible on this resource config"
       ↓ SAT  → Schedule
  → Verilog codegen                (to build)
  → Vivado synthesis + impl        (external)
  → Resource report comparison     (new: parse .rpt files)
```

**What "feasibility checking" means concretely:**
- User specifies resource constraints: `--fpga A.json` with `{DSP: 200, FF: 50000, LUT: 100000}`.
- SDC generator adds resource upper-bound constraints (pool sizes ≤ available DSPs, etc.).
- Solver returns UNSAT → report which constraint is the bottleneck.
- Solver returns SAT → proceed to codegen.

**Sub-tasks:**
1. Define FPGA profile JSON schema.
2. Plumb `--fpga` flag through `main.py` → `SDCGenerator`.
3. Add solver diagnostics: on UNSAT, extract unsatisfiable core and report which resource.
4. Script to run Vivado synthesis from command line (non-GUI batch mode).
5. Script to parse Vivado utilization `.rpt` and compare against profile.

---

### (4) Scheduling Algorithm Revisit  [LOW PRIORITY / FUTURE]

**Concern:** The current Z3-based ASAP/modulo scheduler may not scale to large loop bodies
(e.g., gemm has 4 nested loops with hundreds of operations per II). Z3 SMT can time out.

**Options to consider (when needed):**
- Heuristic pre-solving: ASAP/ALAP bounds to tighten Z3 variable domains.
- Switch inner solver to ILP (CBC, Gurobi) for the scheduling part, keep Z3 for types.
- Iterative relaxation: try II=1, increment until SAT.
- **Not blocking for submission** — flag if Z3 times out on the 3 benchmarks.

---

## Benchmark Characterization

### jacobi-1d (easiest)
- Two sequential 1D stencil passes per time step: `B[i] = 0.33*(A[i-1]+A[i]+A[i+1])`
- Two `Add` + one `Mul` per inner body.
- Outer `t` loop is time, inner `i` loop parallelizable with spaceloop or timeloop.
- **Jalapeno mapping:** outer = `timeloop` over `t`, inner = `timeloop` or `spaceloop` over `i`.
- Recurrence: `A` read-after-write across `t` steps (true WAR/RAW, must pipeline carefully).
- Good first target — no 2D arrays, small II, straightforward constraints.

### mvt (medium)
- Two double loops: `x1[i] += A[i][j]*y1[j]`, `x2[i] += A[j][i]*y2[j]`.
- Each is a dot-product-style accumulation over `j` with `Mult` + `Add` (+ `Reg` for accumulator).
- **Jalapeno mapping:** outer `i` = space or time loop, inner `j` = timeloop with recurrence on `x`.
- Two independent kernels can share DSPs with careful binding.

### gemm (hardest)
- `C[i][j] *= beta` then `C[i][j] += alpha * A[i][k] * B[k][j]` over i,j,k.
- Two multipliers per MAC body, deep recurrence on `C`.
- **Jalapeno mapping:**
  - Outer `i,j` = space loops (partially unrolled) or time loops.
  - Inner `k` = timeloop with recurrence on `C[i][j]`.
  - `alpha * A[i][k]` and `* B[k][j]` chain = 2 mults + 1 add + accumulate.
- Most complex resource allocation — good stress test for feasibility checker.

---

## Vivado Platform Notes

**Can you run Vivado on Mac?**
**No. Vivado does not support macOS.**
Xilinx/AMD only supports Vivado on:
- Linux (RHEL/CentOS/Ubuntu — strongly recommended)
- Windows (limited)

**Recommendation:** Use a Linux server for Vivado synthesis and implementation.
Your Mac development workflow should be:
1. Write + compile `.fil` → `.v` on Mac (all pure Python, works fine).
2. Run tests and feasibility checking on Mac.
3. SSH to Linux server for Vivado synthesis, place-and-route, and bitstream generation.
4. Optionally script the whole flow with a Makefile that SSH's to the server.

**Vivado batch mode** (no GUI needed for automation):
```bash
vivado -mode batch -source run_synth.tcl
```
This is the right mode for scripted end-to-end pipelines.

---

## Immediate Next Steps (ordered)

1. **Nail down the Verilog abstraction** — answer the behavioral vs structural question,
   sketch the module interface for a simple example like `dot_product.fil`.
2. **Implement basic codegen** for flat (non-loop) programs first.
3. **Extend codegen** to handle `timeloop` bodies with II-based pipeline control.
4. **Write `jacobi-1d.fil`** — the simplest benchmark — to shake out both the language
   expressiveness and the codegen.
5. **Run Vivado on jacobi-1d** on the Linux server, verify resource numbers match feasibility prediction.
6. Iterate to `mvt.fil`, then `gemm.fil`.

---

## Open Questions

- What are the two target FPGAs exactly? (Part numbers, resource budgets)
- Fixed-point or floating-point arithmetic? (Affects DSP mapping and latency numbers)
- Should memory (array storage) be on-chip BRAM or off-chip DRAM with AXI interface?
- What's the target clock frequency? (Affects achievable II, pipeline depth assumptions)
- For the submission: is this a conference paper, workshop, or thesis? (Affects how much
  of (4) we need to address)
