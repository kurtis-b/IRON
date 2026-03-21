#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import peak_reference
from peak_reference import (
    build_host_profile,
    build_peak_reference_artifact,
    derive_theoretical_ddr_peak_bytes_per_sec,
    derive_igpu_fp32_peak_ops_per_sec,
    parse_rocm_smi_product_output,
    parse_xrt_smi_examine_json,
    resolve_sku_candidates,
)

LSCPU_HX370 = """\
Architecture:                            x86_64
CPU(s):                                  24
Vendor ID:                               AuthenticAMD
Model name:                              AMD Ryzen AI 9 HX 370 w/ Radeon 890M
CPU family:                              26
Model:                                   36
Thread(s) per core:                      2
Core(s) per socket:                      12
CPU max MHz:                             5157.8950
Flags:                                   avx2 avx512f avx512bw avx512vl avx512_bf16 avx_vnni
"""

CPUINFO_HX370 = """\
processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen AI 9 HX 370 w/ Radeon 890M
flags\t\t: avx2 avx512f avx512bw avx512vl avx512_bf16 avx_vnni
"""

ROCM_SMI_PRODUCT = """\
WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status

{"card0": {"Card Series": "AMD Radeon Graphics", "Card Model": "0x150e", "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]", "Card SKU": "STRIXEMU", "Subsystem ID": "0x38b8", "Device Rev": "0xc1", "Node ID": "1", "GUID": "1487", "GFX Version": "gfx1150"}}
"""

XRT_EXAMINE_JSON = json.dumps(
    {
        "system": {
            "host": {
                "xrt": {
                    "version": "2.21.75",
                    "drivers": [
                        {"name": "virtio-pci", "version": "6.17.0-1012-oem"},
                        {"name": "amdxdna", "version": "6.17.0-1012-oem"},
                    ],
                },
                "devices": [
                    {
                        "bdf": "0000:64:00.1",
                        "name": "RyzenAI-npu4",
                        "firmware_version": "1.1.2.64",
                        "is_ready": "true",
                    }
                ],
            }
        }
    }
)

LSHW_MEMORY_JSON = json.dumps(
    [
        {
            "id": "memory",
            "class": "memory",
            "description": "System memory",
            "size": 33688649728,
        }
    ]
)


def test_parse_rocm_smi_product_output_accepts_warning_prefix():
    parsed = parse_rocm_smi_product_output(ROCM_SMI_PRODUCT)

    assert parsed["detected_model"] == "AMD Radeon Graphics"
    assert parsed["detected_card_sku"] == "STRIXEMU"
    assert parsed["detected_gfx_version"] == "gfx1150"


def test_parse_xrt_smi_examine_json_extracts_npu_fields():
    parsed = parse_xrt_smi_examine_json(XRT_EXAMINE_JSON)

    assert parsed["device_name"] == "RyzenAI-npu4"
    assert parsed["firmware_version"] == "1.1.2.64"
    assert parsed["xrt_version"] == "2.21.75"
    assert parsed["amdxdna_driver_version"] == "6.17.0-1012-oem"


def test_collect_xrt_smi_examine_json_uses_fresh_output_path(tmp_path, monkeypatch):
    class _FakeTempDir:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_run_command(command):
        output_path = Path(command[-1])
        assert output_path == tmp_path / "xrt_examine.json"
        assert not output_path.exists()
        output_path.write_text(XRT_EXAMINE_JSON, encoding="utf-8")
        return ""

    monkeypatch.setattr(
        peak_reference.tempfile, "TemporaryDirectory", lambda: _FakeTempDir()
    )
    monkeypatch.setattr(peak_reference, "_run_command", fake_run_command)

    collected = peak_reference.collect_xrt_smi_examine_json()

    assert (
        json.loads(collected)["system"]["host"]["devices"][0]["name"] == "RyzenAI-npu4"
    )


def test_resolve_sku_candidates_uses_exact_model_name_match():
    resolved = resolve_sku_candidates(
        "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
        cpu_cores=12,
        cpu_threads=24,
    )

    assert resolved["chip_sku"] == "AMD Ryzen AI 9 HX 370"
    assert resolved["codename"] == "Strix Point"
    assert resolved["match_kind"] == "exact_model_name"


def test_resolve_sku_candidates_keeps_ambiguous_hx_family_unknown():
    resolved = resolve_sku_candidates(
        "AMD Ryzen AI 9 HX w/ Radeon 890M",
        cpu_cores=12,
        cpu_threads=24,
    )

    assert resolved["chip_sku"] == "unknown"
    assert resolved["match_kind"] == "unknown"
    assert resolved["candidate_matches"] == [
        "AMD Ryzen AI 9 HX 370",
        "AMD Ryzen AI 9 HX 375",
        "AMD Ryzen AI 9 HX 470",
    ]


def test_build_host_profile_resolves_hx370_and_merges_candidate_defaults():
    profile = build_host_profile(
        lscpu_text=LSCPU_HX370,
        cpuinfo_text=CPUINFO_HX370,
        rocm_smi_text=ROCM_SMI_PRODUCT,
        xrt_smi_json_text=XRT_EXAMINE_JSON,
        lshw_memory_text=LSHW_MEMORY_JSON,
        dmi_info={
            "product_name": "ASUS Vivobook S 15 M5506WA_M5506WA",
            "board_name": "M5506WA",
            "bios_version": "M5506WA.315",
        },
    )

    assert profile["chip_sku"] == "AMD Ryzen AI 9 HX 370"
    assert profile["chip_codename"] == "Strix Point"
    assert profile["sku_match_kind"] == "exact_model_name"
    assert profile["cpu"]["supports_avx512"] is True
    assert profile["cpu"]["supports_avx512_bf16"] is True
    assert profile["igpu"]["model"] == "AMD Radeon 890M"
    assert profile["igpu"]["cu_count"] == 16
    assert profile["npu"]["device_name"] == "RyzenAI-npu4"
    assert profile["npu"]["official_npu_tops"] == 50
    assert profile["memory"]["installed_bytes"] == 33688649728
    assert profile["memory"]["supported_configs"] == [
        {"type": "DDR5", "speed_mt_s": 5600},
        {"type": "LPDDR5X", "speed_mt_s": 8000},
    ]


def test_memory_override_derives_theoretical_ddr_peak():
    profile = build_host_profile(
        lscpu_text=LSCPU_HX370,
        cpuinfo_text=CPUINFO_HX370,
        lshw_memory_text=LSHW_MEMORY_JSON,
        memory_override={
            "memory": {
                "type": "LPDDR5X",
                "speed_mt_s": 8000,
                "interface_bits": 128,
                "source": "manual",
            }
        },
    )

    assert profile["memory"]["type"] == "LPDDR5X"
    assert profile["memory"]["speed_mt_s"] == 8000
    assert profile["memory"]["interface_bits"] == 128
    assert profile["memory"]["source"] == "manual"
    assert profile["memory"]["derived_theoretical_peak_bytes_per_sec"] == pytest.approx(
        128_000_000_000.0
    )


def test_derive_igpu_fp32_peak_ops_per_sec_matches_hx370_seed_reference():
    assert derive_igpu_fp32_peak_ops_per_sec(16, 2900) == pytest.approx(5.9392e12)


def test_derive_theoretical_ddr_peak_bytes_per_sec_uses_mt_s_and_bits():
    assert derive_theoretical_ddr_peak_bytes_per_sec(5600, 128) == pytest.approx(
        89_600_000_000.0
    )


def test_build_peak_reference_artifact_prefers_measured_dtype_matched_peaks():
    host_profile = build_host_profile(
        lscpu_text=LSCPU_HX370,
        cpuinfo_text=CPUINFO_HX370,
        rocm_smi_text=ROCM_SMI_PRODUCT,
        xrt_smi_json_text=XRT_EXAMINE_JSON,
        lshw_memory_text=LSHW_MEMORY_JSON,
        memory_override={
            "memory": {
                "type": "LPDDR5X",
                "speed_mt_s": 8000,
                "interface_bits": 128,
                "source": "manual",
            }
        },
    )

    artifact = build_peak_reference_artifact(
        host_profile,
        {
            "tool_versions": {
                "benchdnn": "3.5.0",
                "rocblas-bench": "6.5.0",
                "stream": "5.10",
            },
            "peaks": {
                "cpu": {
                    "bfloat16": {
                        "measured_peak_ops_per_sec": 1.2e12,
                        "note": "benchdnn large GEMM",
                    },
                    "float32": {
                        "measured_peak_ops_per_sec": 8.5e11,
                    },
                },
                "igpu": {
                    "float32": {
                        "measured_peak_ops_per_sec": 4.9e12,
                    },
                },
                "npu": {
                    "int8": {
                        "measured_peak_ops_per_sec": 4.7e13,
                        "note": "xrt-smi validate --run gemm",
                    },
                    "bfloat16": {
                        "measured_peak_ops_per_sec": 1.7e13,
                    },
                },
            },
            "memory_bandwidth": {
                "copy_bytes_per_sec": 9.8e10,
                "triad_bytes_per_sec": 8.9e10,
                "measured_peak_bytes_per_sec": 8.9e10,
                "note": "STREAM triad",
            },
        },
    )

    rows = {(row["backend"], row["dtype"]): row for row in artifact["references"]}

    assert rows[("cpu", "bfloat16")]["primary_peak_source_kind"] == "measured"
    assert rows[("cpu", "bfloat16")]["primary_peak_ops_per_sec"] == pytest.approx(
        1.2e12
    )
    assert rows[("igpu", "float32")]["primary_peak_source_kind"] == "measured"
    assert rows[("igpu", "float32")][
        "derived_theoretical_peak_ops_per_sec"
    ] == pytest.approx(5.9392e12)
    assert rows[("npu", "int8")]["official_peak_ops_per_sec"] == pytest.approx(5.0e13)
    assert rows[("npu", "int8")]["primary_peak_source_kind"] == "measured"
    assert rows[("npu", "bfloat16")]["primary_peak_source_kind"] == "measured"
    assert artifact["memory_bandwidth"]["primary_peak_source_kind"] == "measured"
    assert artifact["memory_bandwidth"]["primary_peak_bytes_per_sec"] == pytest.approx(
        8.9e10
    )


def test_build_peak_reference_artifact_marks_unmeasured_dtype_sensitive_denominators_unknown():
    host_profile = build_host_profile(
        lscpu_text=LSCPU_HX370,
        cpuinfo_text=CPUINFO_HX370,
        rocm_smi_text=ROCM_SMI_PRODUCT,
        xrt_smi_json_text=XRT_EXAMINE_JSON,
        lshw_memory_text=LSHW_MEMORY_JSON,
    )

    artifact = build_peak_reference_artifact(host_profile)
    rows = {(row["backend"], row["dtype"]): row for row in artifact["references"]}

    assert rows[("cpu", "bfloat16")]["primary_peak_source_kind"] == "unknown"
    assert rows[("cpu", "float32")]["primary_peak_source_kind"] == "unknown"
    assert rows[("igpu", "bfloat16")]["primary_peak_source_kind"] == "unknown"
    assert rows[("igpu", "float16")]["primary_peak_source_kind"] == "unknown"
    assert rows[("npu", "bfloat16")]["primary_peak_source_kind"] == "unknown"
    assert rows[("igpu", "float32")]["primary_peak_source_kind"] == "derived"
    assert rows[("npu", "int8")]["primary_peak_source_kind"] == "official"
    assert artifact["memory_bandwidth"]["primary_peak_source_kind"] == "unknown"
