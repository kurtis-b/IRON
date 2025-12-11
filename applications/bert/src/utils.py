# Copyright (c) Sebastian Raschka under Apache License 2.0.
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import torch
import numpy as np
from ml_dtypes import bfloat16


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        float_arr = tensor.float().detach().cpu().numpy()
        return float_arr.astype(bfloat16)
    return tensor.detach().cpu().numpy()


def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    device = torch.device("cpu")
    if array.dtype == bfloat16:
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16).to(device)
    return torch.from_numpy(array).to(device)


def model_memory_size(model, input_dtype=torch.float32):
    """
    Calculate the estimated memory size of a PyTorch model in gigabytes.

    This function computes the total memory required for the model's parameters,
    gradients, and buffers based on the input data type.

    Args:
        model (torch.nn.Module): The PyTorch model for which to calculate memory size.
        input_dtype (torch.dtype, optional): The data type of the model's input.
                                              Defaults to torch.float32.

    Returns:
        float: The estimated memory size of the model in gigabytes.
    """

    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb


def assign(left, right, tensor_name="unknown"):
    """
    Assigns the value of the right tensor to a new torch.nn.Parameter after validating shape compatibility.

    Parameters:
    left (torch.Tensor or any): The tensor to compare shape with.
    right (torch.Tensor or any): The tensor or value to be assigned.
    tensor_name (str): The name of the tensor for error reporting (default is "unknown").

    Returns:
    torch.nn.Parameter: A new parameter containing the value of right.

    Raises:
    ValueError: If the shapes of left and right do not match.
    """

    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
        )

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))
