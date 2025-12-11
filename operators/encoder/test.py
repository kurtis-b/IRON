# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from operators.encoder.op import AIEBERTEncoder
from operators.encoder.reference import generate_golden_reference
from operators.common.test_utils import compare_tensors


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,num_heads",
    [
        (128, 768, 3072, 12),  # BERT-base configuration
        (64, 512, 2048, 8),  # Smaller configuration for testing
        (256, 1024, 4096, 16),  # Larger configuration
    ],
)
def test_encoder_layer(seq_len, hidden_size, intermediate_size, num_heads):
    """Test BERT encoder layer against golden reference."""

    # Generate golden reference
    reference_data = generate_golden_reference(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        dtype="bf16",
        seed=42,
    )

    # Create AIE encoder operator
    encoder = AIEBERTEncoder(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_aie_columns=4,
    )

    # Set weights from reference (separate Q/K/V weights)
    encoder.q_weight = reference_data["weights"]["q_weight"]
    encoder.k_weight = reference_data["weights"]["k_weight"]
    encoder.v_weight = reference_data["weights"]["v_weight"]
    encoder.attn_output_weight = reference_data["weights"]["attn_output_weight"]
    encoder.ln1_weight = reference_data["weights"]["ln1_weight"]
    encoder.ln1_bias = reference_data["weights"]["ln1_bias"]
    encoder.ffn_up_weight = reference_data["weights"]["ffn_up_weight"]
    encoder.ffn_down_weight = reference_data["weights"]["ffn_down_weight"]
    encoder.ln2_weight = reference_data["weights"]["ln2_weight"]
    encoder.ln2_bias = reference_data["weights"]["ln2_bias"]

    # Run forward pass
    input_tensor = reference_data["input"]
    attention_mask = reference_data["attention_mask"]

    output = encoder.forward(input_tensor, attention_mask)
    expected_output = reference_data["output"]

    # Compare results
    compare_tensors(
        output,
        expected_output,
        rtol=1e-2,
        atol=1e-2,
        name="encoder_output",
    )


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,num_heads",
    [
        (128, 768, 3072, 12),
    ],
)
def test_encoder_layer_shapes(seq_len, hidden_size, intermediate_size, num_heads):
    """Test that encoder layer produces correct output shapes."""

    encoder = AIEBERTEncoder(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_aie_columns=4,
    )

    # Create random input
    input_tensor = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16)

    # Initialize weights with random values (separate Q/K/V weights)
    encoder.q_weight = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16)
    encoder.k_weight = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16)
    encoder.v_weight = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16)
    encoder.attn_output_weight = torch.randn(
        hidden_size, hidden_size, dtype=torch.bfloat16
    )
    encoder.ln1_weight = torch.ones(hidden_size, dtype=torch.bfloat16)
    encoder.ln1_bias = torch.zeros(hidden_size, dtype=torch.bfloat16)
    encoder.ffn_up_weight = torch.randn(
        hidden_size, intermediate_size, dtype=torch.bfloat16
    )
    encoder.ffn_down_weight = torch.randn(
        intermediate_size, hidden_size, dtype=torch.bfloat16
    )
    encoder.ln2_weight = torch.ones(hidden_size, dtype=torch.bfloat16)
    encoder.ln2_bias = torch.zeros(hidden_size, dtype=torch.bfloat16)

    # Run forward pass
    output = encoder.forward(input_tensor)

    # Check output shape
    assert (
        output.shape == input_tensor.shape
    ), f"Expected output shape {input_tensor.shape}, got {output.shape}"


if __name__ == "__main__":
    # Run a simple test
    print("Running BERT Encoder Layer test...")
    test_encoder_layer_shapes(128, 768, 3072, 12)
    print("Test passed!")
