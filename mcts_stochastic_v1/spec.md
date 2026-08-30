# Kernel MCTS Optimizer — Milestone A Specification

## 1. Goal

Build a small, self-contained system that uses Monte Carlo Tree Search (MCTS) to optimize GPU kernels with LLM-generated semantic transformations.

Milestone A implements **CUDA C++ only**, but the software architecture must keep the kernel representation and execution backend abstract enough that CuTe DSL and CUTLASS/CuTe C++ can be added later without changing the MCTS/search logic.

The system should answer the following question:

> Given a correct CUDA C++ kernel, a fixed workload contract, and a fixed LLM-generation budget, can MCTS using semantic optimization strategies find faster valid kernels than simpler LLM-search baselines?

Milestone A is intentionally narrow.

It does **not** include:

- a learned value function,
- a learned policy function,
- AlphaZero-style self-training,
- cross-kernel transfer learning,
- vLLM runtime integration,
- inference-engine generation,
- topology-aware multi-GPU placement,
- runtime kernel dispatch,
- multi-node execution.

Those are later milestones.

The primary output of Milestone A is:

1. a functioning CUDA kernel optimization search system,
2. reproducible search traces,
3. a benchmark suite,
4. comparisons against simpler search baselines,
5. plots of best-found kernel performance versus LLM-generation budget.

---

# 2. High-Level Architecture

The system takes:

- a starting kernel program (CUDA C++ in Milestone A),
- a correctness/workload specification,
- a target GPU,
- a fixed set of semantic optimization strategies,
- an LLM backend.

It repeatedly performs:

```text
root kernel program
      |
      v
MCTS / PUCT selection
      |
      v
semantic optimization strategy
      |
      v
LLM generates candidate CUDA
      |
      v
compile
      |
      v
correctness test
      |
      v
benchmark
      |
      v
new child node
      |
      v
reward backpropagation
```

Profiler information is collected lazily and is used primarily to inform future LLM optimization steps.

---

# 3. Milestone A Success Criterion

The system should compare the following methods under the same LLM-generation budget:

1. Independent random / best-of-N generation
2. Greedy iterative optimization
3. Iterative best-of-K
4. Beam search
5. MCTS with uniform strategy priors
6. MCTS with LLM-provided strategy priors

The main result should be:

```text
best speedup found
vs
number of LLM generations
```

for each benchmark.

The primary resource budget is:

```text
total number of LLM generations
```

not:

- wall-clock tree iterations,
- number of node visits,
- number of profiler runs.

Additional metrics should also be recorded.

---

# 4. Benchmark Suite

Milestone A should use a small mixed suite of approximately six kernels.

The suite should contain both fundamental CUDA kernels and LLM-serving-relevant kernels.

## 4.1 Fundamental Kernels

### 4.1.1 FP16/BF16 GEMM

Purpose:

- tensor-core use,
- tiling,
- shared-memory staging,
- pipelining,
- vectorized loads,
- occupancy/register tradeoffs.

Initial implementation should use a fixed or narrowly constrained shape.

Suggested eventual variants:

- large GEMM,
- skinny/decode-like GEMM.

Do not require one generated implementation to support arbitrary shapes in Milestone A.

### 4.1.2 Softmax / Reduction

Purpose:

- warp-level reductions,
- block-level reductions,
- memory coalescing,
- synchronization,
- occupancy,
- shared-memory optimization.

---

## 4.2 LLM-Serving Kernels

Prefer kernels inspired by or extracted from vLLM where practical.

Suggested initial set:

### 4.2.1 RMSNorm

### 4.2.2 Fused Add + RMSNorm

### 4.2.3 SiLU-and-Multiply

Representative form:

```text
silu(x) * gate
```

### 4.2.4 Q/K RMSNorm + RoPE

This provides a more realistic fused LLM-serving kernel.

Alternative sixth task if implementation complexity is high:

### 4.2.5 RMSNorm -> FP8 Quantization

Either Q/K norm + RoPE or RMSNorm + FP8 quantization may be used initially.

---

## 4.3 Explicitly Out of Scope Initially

Do not start with:

- paged attention,
- full FlashAttention,
- fused MoE grouped GEMM,
- all-to-all MoE communication,
- multi-GPU kernels.

These can be added after the search machinery is validated.

---

# 5. Workload Contract

Every benchmark must define a workload contract.

Conceptually:

```text
WorkloadContract:
    operation
    dtype
    input shapes
    allowed shape range
    tensor layouts
    numerical tolerance
    launch assumptions
    benchmark input distribution
```

A workload contract should be immutable during a search.

For Milestone A, fixed shapes or a small fixed set of shapes are preferred.

If multiple shapes are used, the reward should aggregate across them.

---

# 6. Target GPUs

## 6.1 Primary GPU

Use:

```text
NVIDIA H100 SXM
```

as the primary development target if available.

The benchmark environment must record:

- exact GPU model,
- SXM vs PCIe,
- compute capability,
- driver version,
- CUDA toolkit version,
- nvcc version,
- host system details,
- power limit if available,
- GPU clocks if controlled,
- relevant environment variables.

Do not mix H100 PCIe and H100 SXM measurements in the same result set.

---

## 6.2 Secondary GPU

After the system works on H100, add:

```text
NVIDIA B200
```

as a transfer target.

The purpose is not Milestone A optimization itself, but eventual architecture-transfer experiments.

Initial implementation should assume one target GPU per search run.

---

# 7. Search State / Node Definition

A search node represents a concrete kernel implementation under a fixed environment.

Milestone A uses the `cuda_cpp` backend. Future backends may include `cute_dsl` and `cutlass_cpp` without changing the search-layer semantics.

Conceptually:

```text
State = (
    kernel program,
    backend type,
    workload contract,
    target hardware,
    compiler/toolchain configuration
)
```

The optimization history is **not** part of the state.

Two different search paths may therefore reach the same state.

The implementation may use a DAG / transposition table rather than a strict tree.

Each node should store at least:

```text
Node:
    id
    parent edge references
    program_text
    backend_type
    normalized_program_hash
    binary_hash_or_sass_hash
    compile_status
    correctness_status
    benchmark_results
    reward
    lightweight_profile
    full_profile
    children/action statistics
    created_by_strategy
    created_by_llm_generation_id
```

Profiler information is attached to a node but does not alter kernel semantics.

---

# 8. Action / Edge Definition

An action is a semantic GPU-kernel optimization strategy.

Strategies should be identified by backend-independent semantic IDs. The concrete prompt used for a strategy may differ by backend.

The initial strategy set will contain approximately 22 prompts derived from NVIDIA optimization guidance.

Examples may include:

- improve global-memory coalescing,
- vectorize memory access,
- introduce shared-memory staging,
- change tile dimensions,
- reduce shared-memory traffic,
- reduce bank conflicts,
- reduce synchronization,
- increase occupancy,
- reduce register pressure,
- increase instruction-level parallelism,
- introduce double buffering,
- introduce software pipelining,
- use async copies,
- use TMA where appropriate,
- use tensor cores,
- change warp decomposition,
- apply warp specialization,
- restructure reduction,
- fuse operations,
- reduce redundant loads,
- improve data reuse,
- specialize for workload shape.

The exact strategy prompts should live in configuration files rather than being hard-coded into MCTS logic.

Recommended configuration shape:

```yaml
strategies:
  - id: shared_memory_staging
    description: Improve data reuse using on-chip staging
    prompts:
      cuda_cpp: |
        ...
      cute_dsl: |
        ...
      cutlass_cpp: |
        ...
```

Only `cuda_cpp` prompts are required for Milestone A. Future backend prompt variants should be additive.

---

# 9. Strategy Priors

MCTS uses PUCT.

For a state `s` and semantic action `a`:

```text
score(s, a) =
    Q(s, a)
    +
    c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
```

Where:

```text
P(s, a) = strategy prior
N(s, a) = edge visit count
Q(s, a) = mean backed-up value
```

Milestone A must support at least two prior modes:

### 9.1 Uniform Priors

```text
P(a | s) = 1 / number_of_actions
```

### 9.2 LLM Priors

A separate LLM call may inspect:

- kernel source,
- target hardware,
- workload,
- parent profile summary,

and produce a normalized probability distribution over the available strategies.

The LLM used for priors may be the same model as the generator or a different model.

The implementation should keep these interfaces separate.

---

# 10. Stochastic Action Semantics

A semantic strategy does **not** deterministically produce one child.

Instead:

```text
child_program ~ LLM(
    parent_program,
    backend,
    selected_strategy,
    workload,
    hardware,
    profiler_summary
)
```

Therefore:

```text
(parent state, strategy)
```

may have multiple concrete child implementations.

The tree should conceptually support:

```text
state
  |
  +-- strategy A
  |      |
  |      +-- generated kernel A1
  |      +-- generated kernel A2
  |      +-- generated kernel A3
  |
  +-- strategy B
         |
         +-- generated kernel B1
```

Do not model each strategy as a deterministic edge to exactly one kernel.

---

# 11. Progressive Widening

Do **not** generate K candidate kernels whenever an action is first selected.

Instead, use progressive widening.

Let:

```text
num_children(s, a)
```

be the number of distinct generated kernels currently associated with strategy `a` from state `s`.

A new LLM realization is allowed when:

```text
num_children(s, a)
<
c_pw * N(s, a) ** alpha_pw
```

Suggested initial values:

```text
alpha_pw = 0.5
c_pw = 1.0
K_max = 4
```

`K_max` is the maximum number of concrete LLM-generated children that one semantic action may acquire.

Example behavior:

```text
edge visits = 1   -> approximately 1 child
edge visits = 4   -> approximately 2 children
edge visits = 9   -> approximately 3 children
edge visits = 16  -> approximately 4 children
```

This prevents branch explosion.

---

# 12. Selection Between Existing Realizations

When PUCT selects a semantic action:

### Case A: Progressive widening allows another child

Generate a fresh LLM candidate.

### Case B: The action already has the maximum currently allowed children

Descend into an existing generated child.

A simple UCB-style child selection may be used.

Exact child-selection policy is not a major Milestone A research variable.

Keep it simple and configurable.

---

# 13. LLM Generation Context

Every candidate-generation request should use a **fresh LLM session**.

The generator must be backend-aware, but the MCTS logic must not be.

Do not preserve conversational history from grandparents or siblings.

Provide only:

1. workload/correctness specification,
2. parent kernel program,
3. backend type,
4. target GPU,
5. selected semantic optimization strategy,
6. backend-specific strategy prompt,
7. structured profiler summary for the parent,
8. explicit instruction to preserve semantics,
9. compiler/language constraints if needed.

Conceptually:

```text
P(child | parent, backend, strategy, profile, hardware, workload)
```

The MCTS data structure owns search history.

The LLM conversation does not.

---

# 14. Candidate Repair / Regeneration

A generated candidate may fail to:

- compile,
- launch,
- pass correctness tests.

A failed generation is a failed **realization of a strategy**, not necessarily evidence that the semantic strategy is bad.

For each proposed child:

```text
generate candidate
    |
    v
compile / test
    |
    +-- valid -> benchmark
    |
    +-- invalid
            |
            v
      repair/regenerate
```

Suggested repair budget:

```text
M_repair = 1 or 2
```

Do not spend a large number of LLM calls repairing one candidate.

After the repair budget is exhausted:

```text
proposal_status = INVALID
```

Record:

```text
strategy proposal count
strategy valid proposal count
strategy invalid proposal count
```

Optionally compute:

```text
P_valid(s, a) =
    valid_proposals / total_proposals
```

Do **not** incorporate `P_valid` into PUCT initially.

Just log it for later analysis.

---

# 15. No Rollouts

Milestone A uses no rollout policy.

A newly generated valid leaf is directly evaluated on hardware.

The leaf value is:

```text
compile
-> correctness
-> benchmark
-> measured reward
```

Therefore:

```text
V(leaf) = measured_reward
```

No simulated future trajectory is required.

---

# 16. Correctness

Correctness is a hard feasibility constraint.

Incorrect kernels must never receive a performance reward.

Conceptually:

```text
if not correct:
    candidate = invalid
else:
    reward = performance_reward
```

Correctness tests should include:

- deterministic known cases,
- randomized inputs,
- edge values,
- representative workload shapes,
- numerical tolerance appropriate to dtype.

Where practical, hidden/randomized tests should be generated independently from the LLM prompt.

---

# 17. Performance Reward

Use root-normalized log speedup.

For a single workload shape:

```text
reward(K) = log(T_root / T_K)
```

Where:

```text
T_root = measured latency of initial kernel
T_K = measured latency of candidate kernel
```

Interpretation:

```text
reward = 0      -> equal to root
reward > 0      -> faster than root
reward < 0      -> slower than root
```

For multiple workload shapes:

```text
reward(K) =
    mean_i log(T_root[i] / T_K[i])
```

The benchmark harness should record raw timings as well as reward.

---

# 18. Benchmark Timing

Use CUDA-event-based kernel timing initially.

Each benchmark should include:

```text
warm-up iterations
measurement iterations
median latency
mean latency
standard deviation
minimum latency
possibly p5 / p95
```

Prefer median latency as the primary timing statistic.

Avoid including compilation in kernel runtime.

Compilation cost should be recorded separately.

---

# 19. Evaluation Tiers

The evaluation loop should be hierarchical.

## 19.1 Tier 0 — Mandatory for Every Valid Candidate

Every valid candidate gets:

```text
compile
correctness
warm-up
CUDA-event benchmark
```

This produces the MCTS reward.

---

## 19.2 Tier 1 — Lightweight Profiling

Lightweight profiling is collected **lazily**.

Do not profile every newly generated node.

Instead:

```text
candidate created
    |
    v
compile/test/time
    |
    v
backprop reward
    |
    v
node later selected for expansion
    |
    v
profile if no cached profile exists
```

The purpose of lightweight profiling is to guide the next LLM optimization step.

Conceptually:

```text
timing evaluates nodes
profiling informs expansions
```

The initial implementation may use a small set of metrics such as:

- achieved occupancy,
- DRAM throughput/utilization,
- L2 throughput/hit information,
- tensor-core utilization where relevant,
- shared-memory activity,
- register usage,
- active warps,
- major stall categories,
- instruction count or instruction mix.

The profiler output supplied to the LLM should be summarized into structured text or JSON rather than dumping a huge raw report.

---

## 19.3 Tier 2 — Full Profiling

Full NCU / SASS / deeper profiling should only be collected for selected nodes, for example:

- root kernel,
- new global-best kernels,
- heavily visited nodes,
- debugging cases.

Do not run expensive full profiling for every leaf.

Nsight Systems should not be required in the inner search loop for Milestone A unless the benchmark involves multiple kernels or system-level overlap.

---

# 20. Deduplication

Do not attempt full semantic program equivalence.

Use a simple two-level approach.

## 20.1 Normalized Source Hash

Before compilation:

1. run `clang-format`,
2. strip comments where convenient,
3. normalize whitespace,
4. hash normalized source.

Conceptually:

```text
source_hash = H(normalized_source)
```

This catches exact or formatting-only duplicates.

Variable renaming may still produce different hashes.

That is acceptable.

---

## 20.2 Compiled / SASS Hash

After compilation:

1. disassemble relevant generated code,
2. normalize addresses / metadata / irrelevant symbol noise where practical,
3. hash the instruction representation.

Conceptually:

```text
binary_key = (
    normalized_sass_hash,
    grid,
    block,
    dynamic_shared_memory,
    workload,
    GPU,
    compiler/toolchain
)
```

Two source programs with different local variable names may compile to the same SASS and therefore deduplicate at this stage.

Do not over-engineer SASS normalization initially.

If reliable SASS hashing proves difficult, store binary/cubin hashes first and improve later.

---

# 21. Transposition Table

Search states may be reached through multiple paths.

Use a global transposition table where practical.

Conceptually:

```text
state_key -> canonical node
```

If a newly generated kernel matches an existing state:

```text
reuse existing node
```

but still record the new incoming relationship:

```text
(parent, strategy) -> existing state
```

This creates a DAG rather than a strict tree.

Repeated discovery of the same state is useful data and should not be discarded from logs.

---

# 22. Backpropagation

After a valid leaf is measured:

```text
reward = measured log speedup
```

Backpropagate through selected search edges.

Each semantic action edge tracks:

```text
N(s, a)
W(s, a) = cumulative backed-up value
Q(s, a) = W(s, a) / N(s, a)
```

Initial implementation may use the measured leaf reward directly for all ancestors in the selected path.

No discount factor is necessary unless later experiments show a reason to penalize search depth.

---

# 23. Search Depth

Do not hard-code a very shallow search.

However, impose a configurable maximum depth to prevent pathological sequences.

Suggested initial value:

```text
max_depth = 8 to 12
```

The exact value is less important than the global LLM-generation budget.

A child may be slower than its parent but still lead to a better descendant.

Therefore do not prune merely because:

```text
child_latency > parent_latency
```

PUCT should naturally allocate fewer visits to poor branches.

---

# 24. Stopping Rules

A search terminates when any configured condition is reached.

Primary stopping rule:

```text
total_llm_generations >= generation_budget
```

Optional additional rules:

```text
wall_clock_budget
gpu_evaluation_budget
no_improvement_for_N_generations
max_number_of_nodes
```

For scientific comparisons, LLM-generation count is the primary fixed budget.

---

# 25. Baseline Search Algorithms

All baselines should use the same:

- starting kernel,
- semantic strategy set where relevant,
- LLM backend,
- correctness harness,
- benchmark harness,
- LLM-generation budget.

---

## 25.1 Independent Best-of-N

Every generation starts from the root kernel.

Possible procedure:

```text
for generation in budget:
    choose strategy
    generate candidate from root
    evaluate candidate

return best candidate
```

This tests whether tree structure provides value.

---

## 25.2 Greedy Iterative Search

Maintain one current kernel.

At each step:

```text
generate candidate from current best
if candidate is faster:
    current = candidate
```

Strategy selection may use:

- uniform strategy selection,
- LLM strategy prior.

---

## 25.3 Iterative Best-of-K

At each step:

```text
from current kernel:
    generate K children
    evaluate all
    move to best valid child
```

This is an important strong baseline.

Suggested initial:

```text
K = 4
```

Generation budget must count every candidate.

---

## 25.4 Beam Search

Maintain a beam of top `B_beam` kernels.

At every iteration:

```text
expand beam kernels
evaluate candidates
retain top B_beam
```

Suggested initial beam sizes:

```text
2
4
```

---

## 25.5 MCTS — Uniform Priors

Run MCTS with:

```text
P(a | s) = uniform
```

---

## 25.6 MCTS — LLM Priors

Run MCTS with:

```text
P(a | s) = LLM-provided strategy probabilities
```

This is the main Milestone A system.

---

# 26. Search Budget

Suggested initial budgets per benchmark:

```text
100 LLM generations
250 LLM generations
500 LLM generations
```

The system should support checkpointed measurements so performance can be plotted as a function of budget.

Do not assume 500 is final.

Start smaller during debugging.

---

# 27. Suggested Initial Hyperparameters

```text
num_strategies      ~= 22
K_max               = 4
M_repair             = 2
alpha_pw             = 0.5
c_pw                 = 1.0
max_depth            = 10
generation_budget    = configurable
c_puct               = configurable
```

Do not spend significant effort tuning these before the system works end-to-end.

`c_puct` may require adjustment because its useful scale depends on reward magnitude.

---

# 28. Kernel Backend Abstraction

The search layer must not invoke CUDA-specific compilation, launch, disassembly, or profiling utilities directly.

Define a backend abstraction such as:

```python
class KernelBackend:
    name: str

    def normalize_program(self, program_text: str) -> str:
        ...

    def compile(self, program_text, workload, hardware, toolchain):
        ...

    def benchmark(self, compiled_artifact, workload):
        ...

    def lightweight_profile(self, compiled_artifact, workload):
        ...

    def full_profile(self, compiled_artifact, workload):
        ...

    def binary_fingerprint(self, compiled_artifact, launch_config):
        ...
```

Milestone A implements only:

```text
CudaCppBackend
```

Future implementations may add:

```text
CuTeDSLBackend
CutlassCppBackend
```

without changing:

- MCTS,
- PUCT,
- progressive widening,
- transposition logic,
- reward calculation,
- search baselines,
- strategy-prior logic.

The backend owns backend-specific concerns such as:

```text
program normalization
compiler/JIT invocation
artifact generation
launch configuration extraction
disassembly
profiling commands
binary/SASS fingerprinting
```

Do not implement CuTe DSL or CUTLASS support during Milestone A.

---

# 29. Strategy-Prior Interface

Define an interface such as:

```python
class StrategyPriorProvider:
    def get_priors(
        self,
        kernel_source,
        workload,
        hardware,
        profile_summary,
        strategies,
    ) -> dict[str, float]:
        ...
```

Implementations:

```text
UniformStrategyPrior
LLMStrategyPrior
```

The returned probabilities must sum to 1.

---

# 30. Kernel-Generation Interface

Define an interface such as:

```python
class KernelGenerator:
    def generate(
        self,
        parent_program,
        backend_type,
        strategy,
        workload,
        hardware,
        profile_summary,
    ) -> GeneratedCandidate:
        ...
```

A generation must use a fresh LLM context.

The returned result should include:

```text
raw model output
extracted CUDA source
model metadata
token usage
latency
generation id
prompt hash
```

---

# 31. Backend Compilation Interface

Compilation is invoked through `KernelBackend`.

For Milestone A, `CudaCppBackend.compile(...)` compiles CUDA C++ using the configured CUDA toolchain.

Do not expose a CUDA-specific compiler object to the MCTS layer.

Record:

```text
success
compiler stdout
compiler stderr
compilation time
artifact paths
register information if available
shared-memory usage
binary/cubin/PTX
```

---

# 32. Correctness Interface

Define:

```python
class CorrectnessChecker:
    def check(compiled_kernel, workload) -> CorrectnessResult:
        ...
```

Record:

```text
success
failed test id
maximum error
mean error
reference output metadata
```

Correctness checks should be deterministic when seeded.

---

# 33. Benchmark Interface

Define:

```python
class KernelBenchmarker:
    def benchmark(compiled_kernel, workload) -> BenchmarkResult:
        ...
```

Record:

```text
warmup count
measurement count
individual timings
median latency
mean latency
stddev
min latency
max latency
```

---

# 34. Profiler Interface

Profiling is accessed through `KernelBackend`.

A shared utility may expose:

```python
summarize_for_llm(profile) -> str | dict
```

Profile results must be cached per node/environment.

---

# 35. Repository Structure

Suggested initial repository layout:

```text
kernel-mcts/
|
|-- spec.md
|-- README.md
|-- pyproject.toml
|
|-- configs/
|   |-- strategies.yaml
|   |-- search/
|   |-- benchmarks/
|   `-- hardware/
|
|-- src/
|   |-- search/
|   |   |-- node.py
|   |   |-- edge.py
|   |   |-- mcts.py
|   |   |-- puct.py
|   |   |-- progressive_widening.py
|   |   `-- transposition_table.py
|   |
|   |-- llm/
|   |   |-- generator.py
|   |   |-- priors.py
|   |   `-- prompts.py
|   |
|   |-- backends/
|   |   |-- base.py
|   |   |-- cuda_cpp.py
|   |   `-- dedup.py
|   |
|   |-- evaluation/
|   |   |-- correctness.py
|   |   |-- benchmark.py
|   |   `-- profile_summary.py
|   |
|   |-- baselines/
|   |   |-- independent.py
|   |   |-- greedy.py
|   |   |-- best_of_k.py
|   |   `-- beam.py
|   |
|   |-- workloads/
|   |   |-- base.py
|   |   |-- gemm.py
|   |   |-- reduction.py
|   |   |-- rmsnorm.py
|   |   |-- silu_mul.py
|   |   `-- rope.py
|   |
|   `-- logging/
|       |-- schema.py
|       `-- writer.py
|
|-- benchmarks/
|   |-- gemm/
|   |-- reduction/
|   |-- rmsnorm/
|   |-- fused_add_rmsnorm/
|   |-- silu_mul/
|   `-- qk_norm_rope/
|
|-- scripts/
|   |-- run_search.py
|   |-- run_baseline.py
|   |-- evaluate_best.py
|   `-- plot_results.py
|
`-- tests/
```

This structure is a suggestion, not a hard requirement.

Prefer simple modules over premature abstraction.

---

# 36. Logging Requirements

Logging is important because Milestone B will train from Milestone A search traces.

Every LLM-generated candidate should produce a durable record.

Suggested logical schema:

```text
SearchRun:
    run_id
    benchmark_id
    workload
    hardware
    toolchain
    search_algorithm
    random_seed
    config
    start_time

GenerationRecord:
    generation_id
    run_id
    parent_node_id
    backend_type
    strategy_id
    strategy_prior
    parent_visit_count
    parent_action_Q
    llm_prompt
    prompt_hash
    llm_output
    candidate_program
    repair_attempt
    compile_status
    correctness_status
    timings
    reward
    program_hash
    binary_hash
    profile_summary
    created_node_id

NodeRecord:
    node_id
    program_text
    backend_type
    workload
    hardware
    reward
    benchmark_statistics
    profiler_statistics

EdgeRecord:
    parent_node_id
    backend_type
    strategy_id
    child_node_id
    visit_count
    Q
    prior
```

JSONL or SQLite is sufficient initially.

SQLite is preferred if querying/search reconstruction becomes useful.

---

# 37. Reproducibility

Every run must record enough information to approximately reproduce the result.

Record at least:

```text
git commit
config files
LLM model name/version
LLM request parameters
CUDA version
driver version
nvcc version
GPU model
GPU UUID if appropriate
benchmark random seed
search random seed
compiler flags
strategy prompt versions
backend type
backend implementation version
```

Benchmark noise means bitwise reproducibility is not expected.

Configuration reproducibility is required.

---

# 38. Experimental Metrics

For each benchmark and search algorithm, report:

## Primary

```text
best speedup vs number of LLM generations
```

## Secondary

```text
best absolute latency
final speedup
time to first improvement
number of valid candidates
number of invalid candidates
compile failure rate
correctness failure rate
duplicate rate
GPU benchmark time
profiling time
LLM token usage
LLM wall-clock latency
search depth of best candidate
number of unique states
```

Also record strategy statistics such as:

```text
strategy selection frequency
strategy validity rate
strategy average immediate reward
strategy involvement in best paths
```

These will be useful for Milestone B.

---

# 39. Evaluation Against Expert Implementations

Where available, measure:

```text
generated kernel
vs
starting kernel
vs
expert/reference implementation
```

For GEMM, possible references may include optimized library kernels.

For vLLM-derived operations, use the corresponding production implementation where practical.

The Milestone A search reward should still be normalized against the starting root kernel.

Expert implementations are primarily an external quality reference.

---

# 40. Important Non-Goals

Do not allow the implementation to expand into the following during Milestone A:

```text
policy/value training
distributed MCTS
multi-GPU search
automatic inference-engine generation
topology optimization
cross-hardware transfer
automatic kernel dispatch
general compiler construction
semantic CUDA equivalence checking
large-scale vector databases
complex agent memory systems
```

Keep the search loop understandable and inspectable.

---

# 41. Milestone A Deliverables

Milestone A is complete when all of the following exist.

## 40.1 Functional Search Engine

MCTS can:

```text
load root kernel
select semantic strategy
resolve backend-specific strategy prompt
generate LLM candidate
compile
test
benchmark
create/reuse state
backpropagate reward
continue until budget exhausted
```

## 40.2 Progressive Widening

A strategy may produce multiple LLM realizations without exploding the branching factor.

## 40.3 Lazy Profiling

Profiler data is collected only when useful for expansion.

## 40.4 Deduplication

At minimum:

```text
normalized source hashing
```

Preferably also:

```text
compiled/SASS hashing
```

## 40.5 Baselines

At least:

```text
independent
greedy
best-of-K
MCTS-uniform
MCTS-LLM-prior
```

Beam search may be added shortly afterward.

## 40.6 Benchmark Suite

At least four kernels working end-to-end.

Target six:

```text
GEMM
softmax/reduction
RMSNorm
fused-add RMSNorm
SiLU-and-multiply
Q/K RMSNorm + RoPE
```

## 40.7 Results

Produce plots:

```text
best speedup vs LLM generations
```

across algorithms and benchmarks.

---

# 42. Future Backend Compatibility

The backend abstraction is included now specifically to support later experiments with alternative kernel representations.

Expected path:

```text
CUDA C++
   ->
CuTe DSL
   ->
CUTLASS / CuTe C++
```

Potential future questions include:

```text
Does CuTe DSL reduce invalid-generation rate?
Does a structured kernel language improve search sample efficiency?
Do different backends prefer different semantic strategies?
Can policy/value models transfer across kernel representations?
```

These are not Milestone A requirements.

For CuTe DSL, many high-level semantic strategies should remain valid, but backend-specific prompts will likely refer to:

```text
layouts
tiled copies
MMA atoms
pipelines
TMA
warp specialization
```

For CUTLASS/CuTe C++, some semantic actions may become structured parameter changes rather than free-form LLM rewrites, for example:

```text
tile shape
cluster shape
stage count
mainloop schedule
epilogue schedule
MMA configuration
```

The search system should eventually be able to treat both deterministic backend mutations and stochastic LLM rewrites as proposal mechanisms beneath the same semantic action layer.

Do not implement any of this during Milestone A.

---

# 43. Milestone B Compatibility

Milestone A logging should make it possible to later train AlphaZero-like models.

Preserve training-relevant information:

```text
state
strategy priors
strategy visit counts
Q estimates
selected strategy
generated child
measured reward
validity
profile information
hardware
workload
```

A future policy target may use MCTS visit counts:

```text
pi(a | s) proportional to N(s, a) ** (1 / temperature)
```

A future value model may predict:

```text
expected best achievable descendant performance
```

rather than only current-node latency.

Do not implement this in Milestone A.

Only preserve the data needed for it.

---

# 44. Guiding Principle

Milestone A should remain a search experiment.

The implementation should optimize for:

```text
simplicity
observability
reproducibility
clean search traces
fair generation-budget comparisons
```

The immediate scientific/engineering question is:

> Does structured tree search use a fixed LLM-generation budget more effectively than simpler iterative CUDA optimization strategies?

If the answer is yes, the next milestone is to learn the search policy and value function from the accumulated search traces.
