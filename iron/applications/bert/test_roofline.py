#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from roofline import (
    annotate_benchmark_row,
    dtype_size_bytes,
    estimate_encoder_bytes_per_inference,
    normalize_dtype_name,
)


def test_normalize_dtype_name_accepts_common_aliases():
    assert normalize_dtype_name("torch.bfloat16") == "bfloat16"
    assert normalize_dtype_name("fp16") == "float16"
    assert normalize_dtype_name("float") == "float32"


def test_dtype_size_bytes_maps_supported_types():
    assert dtype_size_bytes("bfloat16") == 2
    assert dtype_size_bytes("float16") == 2
    assert dtype_size_bytes("float32") == 4
    assert dtype_size_bytes("int8") == 1


def test_estimate_encoder_bytes_per_inference_v1_resident_weights():
    model_config = {
        "model_type": "bert",
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "pad_token_id": 0,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }

    estimated = estimate_encoder_bytes_per_inference(
        model_config,
        seq_len=64,
        dtype_bytes=2,
        bytes_model_version="v1",
        weights_policy="resident",
    )

    assert estimated == 19_365_888


def test_estimate_encoder_bytes_per_inference_streamed_weights_exceeds_resident():
    model_config = {
        "model_type": "bert",
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "pad_token_id": 0,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }

    resident = estimate_encoder_bytes_per_inference(
        model_config,
        seq_len=64,
        dtype_bytes=2,
        weights_policy="resident",
    )
    streamed = estimate_encoder_bytes_per_inference(
        model_config,
        seq_len=64,
        dtype_bytes=2,
        weights_policy="streamed",
    )

    assert streamed > resident


def test_annotate_benchmark_row_uses_peak_reference_and_memory_bandwidth():
    model_config = {
        "model_type": "bert",
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "pad_token_id": 0,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }
    row = {
        "mode": "npu",
        "seq_len": 64,
        "dtype": "bfloat16",
        "estimated_flops_per_inference": "1.000000e+09",
        "throughput_flops_per_sec": "5.000000e+10",
    }
    artifact = {
        "chip_sku": "AMD Ryzen AI 9 HX 370",
        "chip_family": "Ryzen AI 9 HX",
        "chip_codename": "Strix Point",
        "references": [
            {
                "backend": "npu",
                "dtype": "bfloat16",
                "primary_peak_ops_per_sec": 4.0e11,
                "primary_peak_source_kind": "measured",
                "peak_source_note": "IRON calibration",
                "official_peak_ops_per_sec": None,
                "derived_theoretical_peak_ops_per_sec": None,
                "measured_peak_ops_per_sec": 4.0e11,
            }
        ],
        "memory_bandwidth": {
            "primary_peak_bytes_per_sec": 1.28e11,
            "primary_peak_source_kind": "measured",
        },
    }

    annotated = annotate_benchmark_row(
        row,
        model_config=model_config,
        peak_reference_artifact=artifact,
    )

    assert annotated["chip_sku"] == "AMD Ryzen AI 9 HX 370"
    assert annotated["primary_peak_source_kind"] == "measured"
    assert annotated["backend_peak_ops_per_sec"] == "4.000000e+11"
    assert annotated["backend_pct_of_peak"] == "0.125000"
    assert annotated["ddr_peak_source_kind"] == "measured"
    assert float(annotated["estimated_bytes_per_inference"]) > 0
    assert float(annotated["operational_intensity_flops_per_byte"]) > 0
    assert float(annotated["roofline_bound_ops_per_sec"]) > 0
    assert float(annotated["roofline_pct"]) > 0


def test_annotate_benchmark_row_leaves_peak_fields_blank_without_artifact():
    model_config = {
        "model_type": "bert",
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "pad_token_id": 0,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }
    row = {
        "mode": "cpu",
        "seq_len": 64,
        "dtype": "bfloat16",
        "estimated_flops_per_inference": "1.000000e+09",
        "throughput_flops_per_sec": "5.000000e+10",
    }

    annotated = annotate_benchmark_row(
        row,
        model_config=model_config,
        peak_reference_artifact=None,
    )

    assert annotated["chip_sku"] == ""
    assert annotated["backend_peak_ops_per_sec"] == ""
    assert annotated["backend_pct_of_peak"] == ""
    assert annotated["ddr_peak_bytes_per_sec"] == ""
    assert annotated["roofline_pct"] == ""
    assert annotated["bytes_model_version"] == "v1"
