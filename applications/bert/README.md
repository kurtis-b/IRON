<!--
SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BERT Sequence Classification

## Model and Tokenizer Download Instructions

To download the necessary files for the model, please follow the links below:

- **Model File**: [model.safetensors](https://huggingface.co/google-bert/bert-base-uncased/tree/main)
- **Config File**: [config.json](https://huggingface.co/google-bert/bert-base-uncased/tree/main)

when calling `inference.py`, you will supply the paths to those two files on the command line, so you can place them anywhere.

## Installation Instructions

Before running `inference.py`, ensure you have the proper environment. To build the environment from scratch, follow the instructions below:

1. Follow the IRON installation instructions in the repository root fist.
   After this, you should have an `ironenv` environment set up and activated.

2. Install the following additional requirements:
   ```
   python3 -m pip install -r applications/bert/requirements_bert.txt
   ```

## Running Inference

Inference with BERT can be run by invoking the `inference.py` script:  
```bash  
cd applications/bert
python3 inference.py /path/to/model.safetensors /path/to/config.json
```

`inference.py` has the following command format:  
```bash
python3 inference.py <weights_file_path> <config_file_path> [--num_samples NUM_SAMPLES] [--fine_tune]
```

### Arguments:
- `weights_file_path`: Path to the weights file (e.g., `model.safetensors`).
- `config_file_path`: Path to the config file (e.g., `config.json`).
- `--num_samples`: (Optional) Set the number of samples to use for evaluation. Default is `10`.
- `--fine_tune`: (Optional) Fine-tune the model before running evaluation.

## AIE Operator Knobs

The BERT app now supports explicit operator-level execution knobs in `aie_config`.

- `use_aie_mha_to_an`: Use `mha_to_an` for MHA + output projection + AddNorm1.
- `use_aie_ffn_addnorm`: Use `ffn_addnorm` (same behavior as legacy `use_aie_addnorm_ffn`).
- `use_aie_encoder_pipeline`: Use the fused `encoder_pipeline` operator.
- `encoder_operator`: Optional selector string (`none`, `mha_to_an`, `ffn_addnorm`, `encoder_pipeline`, `bert_encoder`).

`encoder_operator` is an alias knob; when set, it overrides the corresponding fused-mode booleans.

Example config files:
- `config/config_offload_mha_to_an.json`
- `config/config_offload_addnorm_ffn.json`
- `config/config_offload_encoder_pipeline.json`

Current limitation: `mha_to_an` and `encoder_pipeline` paths are wired for unmasked execution (`attention_mask=None`).
