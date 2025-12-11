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