<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmarking Methodology

This note describes how to benchmark the current BERT encoder paths in this tree and how to interpret comparisons between them.

## Scope

The current benchmark family covers five execution paths:

- CPU `host_hf`
- iGPU `host_hf`
- NPU `encoder_pipeline`
- NPU `gemm_only`
- NPU `operator_runlist`

The main benchmark entrypoints are:

- `cpu_inference.py`
- `igpu_inference.py`
- `npu_inference.py`
- `automated_benchmark.py`

## Shared Benchmark Contract

Unless a run is explicitly configured otherwise, the current methodology is:

- batch size `1`
- deterministic local inputs
- sequence lengths `64,128,256,512,1024,2048,4096,8192`
- host thread count fixed explicitly for fair comparisons
- warmup and timed-run counts controlled from the CLI or unattended job JSON
- results written as one-row child CSVs and one-row-per-case suite CSVs

For lengths above `512`, learned position embeddings are extended by repetition so that host and NPU paths benchmark the same effective backbone shape.

For direct backend comparisons, fix these knobs across all paths:

- model / study id
- `seq_len`
- `num_samples`
- `warmup_runs`
- `runs_per_sample`
- host thread count
- power backend setting

## Benchmark Modes

### `synthetic_dense`

This is the main systems-benchmark mode.

- CPU and iGPU use deterministic synthetic token ids generated from the shared benchmark text path.
- NPU `encoder_pipeline` and `gemm_only` use the same synthetic benchmark input generation.
- NPU `operator_runlist` currently uses a simpler monotonic synthetic token pattern for stability.

Use this mode for:

- backend-to-backend timing
- unattended suites
- roofline / percent-of-peak analysis
- paper-grade systems comparisons

### `model_valid`

This mode uses real tokenizer outputs and attention masks.

- supported today on CPU and iGPU
- currently constrained to `seq_len <= 512`
- not part of the current NPU timing contract

Use it for:

- sanity-checking host-side behavior against more realistic tokenization

### `task_eval`

Reserved and not implemented in the timing harness.

## How Each Path Executes

This section describes the execution model, not just the benchmark CLI.

### CPU `host_hf`

- Loads the local Hugging Face encoder model on CPU.
- Runs embeddings, attention, FFN, residuals, and layer norms on the host in one framework forward path.
- Uses the shared benchmark sample generator and measures one timed model forward per inference.
- Has no topology selection, no NPU compilation step, and no device-specific dispatch metadata.

What this path measures:

- end-to-end host execution of the canonical encoder model under the selected dtype and thread count

### iGPU `host_hf`

- Loads the same local Hugging Face encoder model on a GPU-enabled PyTorch runtime.
- Runs the full encoder stack on the iGPU rather than splitting work between host and NPU kernels.
- Uses the same benchmark sample generator as CPU and measures one timed framework forward per inference.
- Has no NPU topology policy or runlist structure.

What this path measures:

- end-to-end framework execution on the iGPU, with host orchestration but no per-op NPU dispatch decomposition

### NPU `encoder_pipeline`

- Runs embeddings on the host-side local backbone wrapper.
- Casts the embedding output to the configured NPU runtime dtype.
- Selects one fused `encoder_pipeline` topology per sequence length.
- Compiles and prepares the fused operator for that topology, then runs the encoder through that fused path.
- Reports staged timings for embeddings, QKV projection, and fused encoder-pipeline time.

Important execution details:

- this is the most fused current NPU path
- topology selection changes how sequence tiles, head parallelism, and FFN parallelism are mapped
- `fixed`, `cache`, and `autotune` affect the setup path, but the steady-state latency is measured after the topology is chosen

What this path measures:

- fused NPU execution with minimal host-side orchestration between encoder sub-stages

### NPU `gemm_only`

- Runs embeddings on the host-side local backbone wrapper.
- Offloads only GEMM-shaped work to the NPU:
  - QKV projection
  - attention score GEMMs
  - attention output GEMMs
  - output projection
  - FFN up projection
  - FFN down projection
- Keeps non-GEMM work on the host:
  - head reshaping / layout transforms
  - softmax
  - residual adds
  - layer norms
  - GELU

Important execution details:

- the current implementation binds the six GEMM workloads in a layer to one shared runtime `xclbin`, with workload-specific instruction binaries
- dispatch count scales with attention heads: `4 + 2 * num_heads` per layer in the current implementation
- the benchmark records host preprocess, NPU GEMM, and host postprocess timing separately

What this path measures:

- a partially offloaded NPU system where matrix multiplies move to the device but orchestration and nonlinearities remain on the host

### NPU `operator_runlist`

- Runs embeddings on the host-side local backbone wrapper.
- Reuses one stitched `AIEBERTEncoder` runlist-backed operator across all transformer layers.
- For each layer, the wrapper assigns that layer's weights into the shared encoder operator and dispatches the stitched runlist once.
- Forces no-bias execution in the current benchmark contract.

Important execution details:

- this is the least fused current benchmarked NPU path
- it is run as one sequence length per child process in the unattended harness
- the harness invokes it through `npu_inference_import_main.py`
- the current benchmark records `operator_runlist` time plus dispatch and instruction-binary counts

What this path measures:

- a lower-fusion stitched NPU execution path where orchestration cost is more visible than in `encoder_pipeline`

## Path-Specific Methodology

### CPU `host_hf`

Path:

- Hugging Face encoder on CPU
- current default dtype is `bfloat16`

Methodology:

- use `cpu_inference.py`
- keep `--num-threads` fixed across comparison runs
- use `synthetic_dense` for backend timing studies
- use `model_valid` only when you intentionally want tokenizer-driven host validation

Notes:

- CPU can optionally run with `--disable-all-biases`
- no-bias CPU runs are mainly for parity-aligned comparisons against current NPU paths

### iGPU `host_hf`

Path:

- Hugging Face encoder on ROCm / GPU-enabled PyTorch
- current default dtype is `bfloat16`

Methodology:

- use `igpu_inference.py`
- keep `--num-threads` fixed just like CPU runs
- keep `--device-index` fixed
- use `synthetic_dense` for timing comparisons

Notes:

- iGPU can optionally run with `--disable-all-biases`
- like CPU, no-bias iGPU runs are mainly for parity-aligned studies

### NPU `encoder_pipeline`

Path:

- embeddings plus the fused `encoder_pipeline` operator
- topology-sensitive execution

Methodology:

- use `npu_inference.py --execution-mode encoder_pipeline`
- benchmark only in `synthetic_dense`
- use `--topology-policy cache` for steady-state comparisons
- warm or regenerate the topology cache before important runs
- treat autotune and cache-miss time separately from steady-state latency

Recommended comparison practice:

- use `cache` for the mainline dataset
- keep the selected topology ids recorded in the CSV
- use `autotune` only to build or refresh the cache

### NPU `gemm_only`

Path:

- GEMM-shaped work on the NPU
- non-GEMM work remains on the host

Methodology:

- use `npu_inference.py --execution-mode gemm_only`
- benchmark in `synthetic_dense`
- keep host thread count fixed because host-side work is still material
- compare it against `encoder_pipeline` to isolate the value of fusion

Interpretation:

- this path measures a partially offloaded system
- stage fields such as host preprocess / host postprocess / NPU GEMM latency matter

### NPU `operator_runlist`

Path:

- stitched encoder-operator runlist path
- least fused current NPU execution mode

Methodology:

- use `npu_inference.py --execution-mode operator_runlist`
- benchmark in `synthetic_dense`
- one sequence length per child process
- run it through `npu_inference_import_main.py` when invoked from the unattended harness

Current contract:

- `operator_runlist` forces no-bias execution
- that means it is not directly apples-to-apples with default bias-enabled CPU / iGPU timing runs unless you intentionally disable biases there too

Interpretation:

- compare it primarily as a lower-fusion NPU orchestration point
- validate parity separately with the no-bias contract before drawing correctness-sensitive conclusions

## Parity Methodology

Use `validate_npu_parity.py` for CPU-vs-NPU hidden-state checks.

Current shared default tolerance window:

- `rtol = 0.04`
- `atol = 0.15`
- `max-error-fraction = 0.05`

Current intended use:

- CPU no-bias vs NPU `encoder_pipeline`
- CPU no-bias vs NPU `gemm_only`
- CPU no-bias vs NPU `operator_runlist`

If you mix bias-enabled timing runs with no-bias parity results, call that out explicitly.

## Unattended Methodology

Use `automated_benchmark.py` directly or `run_automated_benchmark_job.py` through a JSON job file.

The unattended harness now:

- fans out NPU cases across `--npu-execution-modes`
- checkpoints completed case ids in a state JSON
- writes one child CSV per case plus one suite CSV
- resumes incomplete runs from the saved state
- retries transient `operator_runlist` child failures that abort before writing a child CSV

Recommended unattended practice:

- use `power_backend none` for fast smoke validation
- enable power logging only after the case matrix is stable
- keep `encoder_pipeline` on `cache` for mainline runs
- verify a short supervised smoke before enabling reboot/resume

## Recommended Run Types

### Smoke Run

Use this to validate functionality quickly:

- `num_samples = 1`
- `warmup_runs = 0`
- `runs_per_sample = 1`
- `power_backend = none`

### Comparison Run

Use this to compare backends before final publication:

- fixed host thread count
- fixed model and sequence set
- same `num_samples`, `warmup_runs`, and `runs_per_sample`
- `power_backend = none` unless power is part of the question
- `encoder_pipeline` on warmed `cache`

### Publishable Run

Use this only after smoke and comparison runs are stable:

- fixed peak-reference artifact
- fixed bytes-model policy
- fixed topology cache for `encoder_pipeline`
- recorded parity results for any no-bias comparisons
- no ad hoc retries outside the harness contract

## Practical Caveats

- `operator_runlist` is currently the least stable backend and should always be validated with a small supervised or unattended smoke before a long run.
- `encoder_pipeline` is the only current NPU path that uses topology selection.
- `gemm_only` and `operator_runlist` should be interpreted as lower-fusion comparison points, not just slower versions of the same implementation.
- When the question is paper-grade fairness, explicitly state whether the comparison is:
  - default backend contract
  - or bias-aligned no-bias contract
