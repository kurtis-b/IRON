#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from benchmark_power import ACTIVE_POWER_FIELDNAMES

SAMPLE_TEXT = """
SCENE I. King Lear's palace.
Enter KENT, GLOUCESTER, and EDMUND
KENT
I thought the king had more affected the Duke of Albany than Cornwall.
GLOUCESTER
It did always seem so to us: but now, in the division of the kingdom, it appears not
which of the dukes he values most; for equalities are so weighed, that curiosity in
neither can make choice of either's moiety. The king is coming. The lords of France
and Burgundy wait in court. We have divided in three our kingdom, and we would know
which daughter loves us most. Sir, I love you more than words can wield the matter;
dearer than eyesight, space, and liberty; beyond what can be valued, rich or rare.
Nothing will come of nothing: speak again. I love your majesty according to my bond,
nor more nor less. Good my lord, you have begot me, bred me, loved me: I return those
duties back as are right fit, obey you, love you, and most honour you.
"""

CSV_FIELDNAMES = [
    "study_id",
    "benchmark_mode",
    "execution_mode",
    "seq_len",
    "num_threads",
    "dtype",
    "num_samples",
    "runs_per_sample",
    "warmup_runs",
    "measured_inference_count",
    "timed_total_sec",
    "throughput_inferences_per_sec",
    "model_type",
    "shape",
    "estimated_flops_per_inference",
    "throughput_flops_per_sec",
    "topology_id",
    "parallel_seq",
    "parallel_heads",
    "parallel_ffn",
    "compute_tile_count",
    "compute_tile_utilization_fraction",
    "compile_setup_time_ms",
    "topology_selection_time_ms",
    "topology_cache_status",
    "cached_steady_state_avg_latency_ms",
    "avg_embedding_latency_ms",
    "avg_qkv_projection_latency_ms",
    "avg_encoder_pipeline_latency_ms",
    "min_latency_ms",
    "avg_latency_ms",
    "max_latency_ms",
] + ACTIVE_POWER_FIELDNAMES

DEFAULT_BENCHMARK_SEQ_LENS = "64,128,256,512,1024,2048,4096,8192"
DEFAULT_BENCHMARK_MODE = "synthetic_dense"
SUPPORTED_BENCHMARK_MODES = ("synthetic_dense", "model_valid", "task_eval")
MODEL_VALID_MAX_SEQ_LEN = 512


def _detect_physical_cores_from_sysfs(respect_affinity=True):
    cpu_root = Path("/sys/devices/system/cpu")
    if not cpu_root.exists():
        return None

    allowed_cpus = None
    if respect_affinity:
        try:
            allowed_cpus = os.sched_getaffinity(0)
        except AttributeError:
            allowed_cpus = None

    physical_cores = set()
    for cpu_dir in cpu_root.glob("cpu[0-9]*"):
        cpu_id = int(cpu_dir.name[3:])
        if allowed_cpus is not None and cpu_id not in allowed_cpus:
            continue
        online_file = cpu_dir / "online"
        if online_file.exists() and online_file.read_text().strip() == "0":
            continue
        core_id_file = cpu_dir / "topology" / "core_id"
        package_id_file = cpu_dir / "topology" / "physical_package_id"
        if not (core_id_file.exists() and package_id_file.exists()):
            continue
        core_id = int(core_id_file.read_text().strip())
        package_id = int(package_id_file.read_text().strip())
        physical_cores.add((package_id, core_id))

    return len(physical_cores) if physical_cores else None


def _detect_physical_cores_from_lscpu(respect_affinity=True):
    try:
        output = subprocess.check_output(
            ["lscpu", "-p=CPU,CORE,SOCKET"], text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    allowed_cpus = None
    if respect_affinity:
        try:
            allowed_cpus = os.sched_getaffinity(0)
        except AttributeError:
            allowed_cpus = None

    physical_cores = set()
    for line in output.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu_id, core_id, socket_id = (int(value) for value in line.split(","))
        if allowed_cpus is not None and cpu_id not in allowed_cpus:
            continue
        physical_cores.add((socket_id, core_id))

    return len(physical_cores) if physical_cores else None


def detect_physical_core_count():
    detected = _detect_physical_cores_from_sysfs(respect_affinity=True)
    if detected:
        return detected
    detected = _detect_physical_cores_from_lscpu(respect_affinity=True)
    if detected:
        return detected
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def detect_max_physical_core_count():
    detected = _detect_physical_cores_from_sysfs(respect_affinity=False)
    if detected:
        return detected
    detected = _detect_physical_cores_from_lscpu(respect_affinity=False)
    if detected:
        return detected
    return detect_physical_core_count()


def detect_logical_thread_count():
    try:
        allowed_cpus = os.sched_getaffinity(0)
        if allowed_cpus:
            return len(allowed_cpus)
    except AttributeError:
        pass
    return os.cpu_count() or 1


def configure_cpu_thread_env(num_threads):
    thread_count = str(num_threads)
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[env_var] = thread_count
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def add_cooldown_args(parser):
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=0.0,
        help="Optional fixed wait inserted before each benchmark segment.",
    )
    parser.add_argument(
        "--cooldown-until-temp-c",
        type=float,
        default=None,
        help=(
            "Optional temperature threshold. When set, wait until the selected "
            "temperature source drops to or below this value before each benchmark "
            "segment."
        ),
    )
    parser.add_argument(
        "--cooldown-temp-source",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help=(
            "Temperature source family used by --cooldown-until-temp-c. auto picks "
            "CPU/package sensors for cpu/npu and GPU sensors for igpu."
        ),
    )
    parser.add_argument(
        "--cooldown-temp-tolerance-frac",
        type=float,
        default=0.05,
        help=(
            "Relative slack above --cooldown-until-temp-c that is still accepted "
            "as cooled enough. For example, 0.05 with a 50 C threshold allows the "
            "benchmark to proceed at or below 52.5 C."
        ),
    )
    parser.add_argument(
        "--cooldown-poll-sec",
        type=float,
        default=2.0,
        help="Polling interval while waiting for a temperature threshold.",
    )
    parser.add_argument(
        "--cooldown-timeout-sec",
        type=float,
        default=300.0,
        help=(
            "Maximum time to wait for the temperature target before continuing "
            "with the benchmark anyway."
        ),
    )


def add_benchmark_mode_args(parser):
    parser.add_argument(
        "--benchmark-mode",
        choices=SUPPORTED_BENCHMARK_MODES,
        default=DEFAULT_BENCHMARK_MODE,
        help=(
            "Benchmark workload mode. synthetic_dense uses deterministic synthetic "
            "token ids. model_valid uses a real local tokenizer and attention masks. "
            "task_eval is reserved for task-level evaluation and is not implemented "
            "in the timing CLIs yet."
        ),
    )


def benchmark_mode_case_suffix(benchmark_mode):
    if benchmark_mode == DEFAULT_BENCHMARK_MODE:
        return ""
    return f"_{benchmark_mode}"


def default_execution_mode_for_backend(backend_mode):
    return {
        "cpu": "host_hf",
        "igpu": "host_hf",
        "npu": "encoder_pipeline",
    }[backend_mode]


def validate_benchmark_mode_request(benchmark_mode, *, backend_mode, seq_lens):
    if benchmark_mode == "task_eval":
        raise NotImplementedError(
            "benchmark-mode task_eval is reserved for task-level evaluation scripts "
            "and is not implemented in the timing CLIs yet"
        )
    if benchmark_mode == "model_valid":
        if any(int(seq_len) > MODEL_VALID_MAX_SEQ_LEN for seq_len in seq_lens):
            raise ValueError(
                f"benchmark-mode model_valid only supports seq_len <= {MODEL_VALID_MAX_SEQ_LEN}"
            )
        if backend_mode == "npu":
            raise ValueError(
                "benchmark-mode model_valid is not supported for npu because the "
                "encoder_pipeline path currently requires attention_mask=None"
            )


def summarize_latency_measurements(latencies_ms, estimated_flops_per_inference):
    if not latencies_ms:
        raise ValueError("Expected at least one latency sample")

    timed_total_sec = sum(latencies_ms) / 1000.0
    measured_inference_count = len(latencies_ms)
    throughput_inferences_per_sec = (
        measured_inference_count / timed_total_sec if timed_total_sec > 0 else 0.0
    )
    throughput_flops_per_sec = (
        (estimated_flops_per_inference * measured_inference_count) / timed_total_sec
        if timed_total_sec > 0
        else 0.0
    )
    return {
        "measured_inference_count": measured_inference_count,
        "timed_total_sec": timed_total_sec,
        "throughput_inferences_per_sec": throughput_inferences_per_sec,
        "estimated_flops_per_inference": estimated_flops_per_inference,
        "throughput_flops_per_sec": throughput_flops_per_sec,
        "min_latency_ms": min(latencies_ms),
        "avg_latency_ms": sum(latencies_ms) / measured_inference_count,
        "max_latency_ms": max(latencies_ms),
    }


def parse_seq_lens(seq_lens_arg):
    seq_lens = []
    for token in seq_lens_arg.split(","):
        token = token.strip()
        if not token:
            continue
        seq_len = int(token)
        if seq_len <= 0:
            raise ValueError(f"Sequence lengths must be positive (got {seq_len})")
        seq_lens.append(seq_len)
    if not seq_lens:
        raise ValueError("At least one sequence length must be provided")
    return seq_lens


def _load_sensors_json():
    result = subprocess.run(
        ["sensors", "-j"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to read sensors JSON\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse sensors JSON output: {exc}") from exc


def _sensor_temp_candidates():
    candidates = []
    for chip_name, chip_data in _load_sensors_json().items():
        if not isinstance(chip_data, dict):
            continue
        for label, values in chip_data.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if not key.startswith("temp") or not key.endswith("_input"):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                candidates.append(
                    {
                        "chip": chip_name,
                        "label": label,
                        "key": key,
                        "source": f"{chip_name}:{label}:{key}",
                        "temperature_c": float(value),
                    }
                )
    return candidates


def _pick_temp_candidate(candidates, preference):
    if preference == "gpu":
        selectors = (
            lambda c: c["chip"].startswith("amdgpu") and c["label"] == "edge",
            lambda c: c["chip"].startswith("amdgpu") and c["label"] == "junction",
            lambda c: c["chip"].startswith("amdgpu"),
            lambda c: c["chip"].startswith("k10temp") and c["label"] == "Tctl",
            lambda c: c["chip"].startswith("acpitz"),
        )
    else:
        selectors = (
            lambda c: c["chip"].startswith("k10temp") and c["label"] == "Tctl",
            lambda c: c["chip"].startswith("k10temp"),
            lambda c: c["chip"].startswith("acpitz"),
            lambda c: c["chip"].startswith("amdgpu") and c["label"] == "edge",
            lambda c: c["chip"].startswith("amdgpu"),
        )

    for selector in selectors:
        for candidate in candidates:
            if selector(candidate):
                return candidate
    return candidates[0] if candidates else None


def read_temperature_reading(mode, requested_source):
    preference = requested_source
    if preference == "auto":
        preference = "gpu" if mode == "igpu" else "cpu"
    candidate = _pick_temp_candidate(_sensor_temp_candidates(), preference)
    if candidate is None:
        raise RuntimeError("No temperature sensors are available for cooldown polling")
    return candidate


def acceptable_cooldown_temp(threshold_c, tolerance_frac):
    threshold_c = float(threshold_c)
    tolerance_frac = max(0.0, float(tolerance_frac))
    return threshold_c * (1.0 + tolerance_frac)


def cooldown_before_benchmark(args, *, mode, label):
    if args.cooldown_sec <= 0 and args.cooldown_until_temp_c is None:
        return {
            "cooldown_wait_sec": 0.0,
            "cooldown_start_temp_c": None,
            "cooldown_end_temp_c": None,
            "cooldown_temp_source": "",
        }

    start_time = time.perf_counter()
    start_temp_c = None
    end_temp_c = None
    temp_source = ""

    if args.cooldown_until_temp_c is not None:
        reading = read_temperature_reading(mode, args.cooldown_temp_source)
        start_temp_c = reading["temperature_c"]
        temp_source = reading["source"]
        accepted_temp_c = acceptable_cooldown_temp(
            args.cooldown_until_temp_c,
            args.cooldown_temp_tolerance_frac,
        )
        print(
            f"Cooldown before {label}: waiting for temperature <= "
            f"{accepted_temp_c:.1f} C from {temp_source} "
            f"(target {args.cooldown_until_temp_c:.1f} C, "
            f"tolerance {args.cooldown_temp_tolerance_frac * 100.0:.1f}%)",
            flush=True,
        )

    if args.cooldown_sec > 0:
        print(
            f"Cooldown before {label}: sleeping for {args.cooldown_sec:.3f} sec",
            flush=True,
        )
        time.sleep(args.cooldown_sec)

    if args.cooldown_until_temp_c is not None:
        deadline = time.perf_counter() + args.cooldown_timeout_sec
        accepted_temp_c = acceptable_cooldown_temp(
            args.cooldown_until_temp_c,
            args.cooldown_temp_tolerance_frac,
        )
        while True:
            reading = read_temperature_reading(mode, args.cooldown_temp_source)
            temp_source = reading["source"]
            end_temp_c = reading["temperature_c"]
            if end_temp_c <= accepted_temp_c:
                break
            if time.perf_counter() >= deadline:
                print(
                    f"Cooldown before {label}: continuing after "
                    f"{args.cooldown_timeout_sec:.1f} sec even though {temp_source} "
                    f"is still {end_temp_c:.1f} C (accepted <= {accepted_temp_c:.1f} C)",
                    flush=True,
                )
                break
            time.sleep(max(0.1, args.cooldown_poll_sec))

    wait_sec = time.perf_counter() - start_time
    return {
        "cooldown_wait_sec": wait_sec,
        "cooldown_start_temp_c": start_temp_c,
        "cooldown_end_temp_c": end_temp_c,
        "cooldown_temp_source": temp_source,
    }


def build_benchmark_texts(num_samples):
    lines = [line.strip() for line in SAMPLE_TEXT.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Benchmark text corpus is empty")

    texts = []
    for sample_idx in range(num_samples):
        rotate = sample_idx % len(lines)
        rotated = lines[rotate:] + lines[:rotate]
        texts.append(" ".join(rotated))
    return texts


def encode_benchmark_texts(texts, seq_len, vocab_size, pad_token_id=0):
    import torch

    token_pattern = re.compile(r"[A-Za-z0-9']+")
    encoded_samples = []
    for text in texts:
        words = token_pattern.findall(text.lower())
        if not words:
            words = ["[unk]"]

        body_tokens = []
        for word in words:
            digest = hashlib.sha1(word.encode("utf-8")).hexdigest()
            token_id = 103 + (int(digest[:8], 16) % max(1, vocab_size - 103))
            body_tokens.append(token_id)

        body_len = max(seq_len - 2, 0)
        if body_len > 0:
            repeats = (body_len + len(body_tokens) - 1) // len(body_tokens)
            body_tokens = (body_tokens * repeats)[:body_len]
        else:
            body_tokens = []

        token_ids = [101]
        if seq_len > 1:
            token_ids.extend(body_tokens)
            token_ids.append(102)
        token_ids = token_ids[:seq_len]
        if len(token_ids) < seq_len:
            token_ids.extend([pad_token_id] * (seq_len - len(token_ids)))

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        encoded_samples.append(
            {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
            }
        )
    return encoded_samples


def _tensorize_tokenizer_field(values):
    import torch

    if hasattr(values, "shape"):
        tensor = values
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(dtype=torch.long)
    if values and isinstance(values[0], (list, tuple)):
        return torch.tensor(values, dtype=torch.long)
    return torch.tensor([values], dtype=torch.long)


def load_local_benchmark_tokenizer(weights_file_path, config_file_path):
    from transformers import AutoTokenizer

    candidate_dirs = []
    for path_like in (weights_file_path, config_file_path):
        parent = Path(path_like).resolve().parent
        if parent not in candidate_dirs:
            candidate_dirs.append(parent)

    errors = []
    for candidate_dir in candidate_dirs:
        try:
            return AutoTokenizer.from_pretrained(
                str(candidate_dir),
                local_files_only=True,
                use_fast=True,
            )
        except Exception as exc:
            errors.append(f"{candidate_dir}: {exc}")

    checked = ", ".join(str(path) for path in candidate_dirs)
    raise RuntimeError(
        "benchmark-mode model_valid requires local tokenizer assets. "
        f"Checked: {checked}. Re-run download_model.py for the study or place "
        "tokenizer files next to the local model weights."
        + (f" Errors: {' | '.join(errors)}" if errors else "")
    )


def encode_model_valid_texts(texts, seq_len, tokenizer):
    encoded_samples = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=int(seq_len),
            return_attention_mask=True,
        )
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise RuntimeError(
                "Tokenizer output must include input_ids and attention_mask for "
                "benchmark-mode model_valid"
            )
        input_ids = _tensorize_tokenizer_field(encoded["input_ids"])
        attention_mask = _tensorize_tokenizer_field(encoded["attention_mask"])
        token_type_values = encoded.get("token_type_ids")
        if token_type_values is None:
            token_type_ids = input_ids.new_zeros(input_ids.shape)
        else:
            token_type_ids = _tensorize_tokenizer_field(token_type_values)
        encoded_samples.append(
            {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }
        )
    return encoded_samples


def prepare_benchmark_samples(
    *,
    benchmark_mode,
    texts,
    seq_len,
    vocab_size,
    pad_token_id,
    weights_file_path,
    config_file_path,
):
    if benchmark_mode == "synthetic_dense":
        return encode_benchmark_texts(
            texts=texts,
            seq_len=seq_len,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
        )
    if benchmark_mode == "model_valid":
        tokenizer = load_local_benchmark_tokenizer(
            weights_file_path,
            config_file_path,
        )
        return encode_model_valid_texts(
            texts=texts,
            seq_len=seq_len,
            tokenizer=tokenizer,
        )
    if benchmark_mode == "task_eval":
        raise NotImplementedError(
            "benchmark-mode task_eval is reserved for task-level evaluation scripts "
            "and is not implemented in the timing CLIs yet"
        )
    raise ValueError(f"Unsupported benchmark_mode {benchmark_mode!r}")


def write_results_csv(output_csv, rows):
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
