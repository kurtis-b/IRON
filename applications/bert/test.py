#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import pytest
from pathlib import Path

test_dir = Path(__file__).parent
weights_dir = Path(f"{test_dir}/bert_base")


def generate_test_params():
    prompt_lengths = [512, 1024, 2048]
    num_samples_list = [100]
    # TODO: Removed fused MHA from tests because it looks like the kernel is doing causal masking,
    # isn't done for the inference runs here
    configs_list = [
        "",
        "_offload_separate",
        "_offload_ffn",
        "_offload_addnorm",
        "_offload_addnorm_ffn",
        "_offload_encoder",
        # "_offload_mha",
        # "_offload_mha_ffn",
        # "_offload_mha_addnorm_ffn",
    ]

    params = []
    names = []
    for prompt_len in prompt_lengths:
        for num_samples in num_samples_list:
            for config in configs_list:
                params.append((prompt_len, num_samples, config))
                name = f"bert_p_{prompt_len}_s_{num_samples}"
                if config != "":
                    name += f"_cfg{config}"
                names.append(name)
    return params, names


params, names = generate_test_params()


@pytest.mark.metrics(
    Latency=r"Average Inference Time per Sample: (?P<value>[\d\.e\+-]+) milliseconds",
)
@pytest.mark.parametrize("prompt_len,num_samples,config", params, ids=names)
def test_llama_3_2_1b(prompt_len, num_samples, config):
    command = f"python3 {test_dir}/inference.py {weights_dir}/model.safetensors {test_dir}/config/config{config}.json --num_samples {num_samples} --seq_len {prompt_len}"

    result = subprocess.run(
        command,
        cwd=test_dir,
        shell=True,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert (
        result.returncode == 0
    ), f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"

    print(result.stdout)
