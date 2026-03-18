#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import os
import re
import subprocess
from pathlib import Path

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
    "seq_len",
    "num_threads",
    "dtype",
    "num_samples",
    "runs_per_sample",
    "warmup_runs",
    "model_type",
    "shape",
    "topology_id",
    "parallel_seq",
    "parallel_heads",
    "parallel_ffn",
    "min_latency_ms",
    "avg_latency_ms",
    "max_latency_ms",
]


def _detect_physical_cores_from_sysfs():
    cpu_root = Path("/sys/devices/system/cpu")
    if not cpu_root.exists():
        return None

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


def _detect_physical_cores_from_lscpu():
    try:
        output = subprocess.check_output(
            ["lscpu", "-p=CPU,CORE,SOCKET"], text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

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
    detected = _detect_physical_cores_from_sysfs()
    if detected:
        return detected
    detected = _detect_physical_cores_from_lscpu()
    if detected:
        return detected
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


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


def write_results_csv(output_csv, rows):
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
