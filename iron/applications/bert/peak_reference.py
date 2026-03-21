#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_VERSION = "1"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "host_profile.json"
DEFAULT_PEAK_REFERENCE_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "peak_references.json"
)
SKU_MATCH_UNKNOWN = "unknown"

AMD_HX_AI_9_SKU_CANDIDATES = (
    {
        "chip_sku": "AMD Ryzen AI 9 HX 370",
        "chip_family": "Ryzen AI 9 HX",
        "codename": "Strix Point",
        "cpu": {
            "cores": 12,
            "threads": 24,
            "max_boost_ghz": 5.1,
            "zen5_cores": 4,
            "zen5c_cores": 8,
        },
        "igpu": {
            "model": "AMD Radeon 890M",
            "cu_count": 16,
            "gfx_freq_mhz": 2900,
        },
        "npu": {
            "official_npu_tops": 50,
            "official_overall_tops": 80,
        },
        "memory": {
            "channels": 2,
            "supported_configs": (
                {"type": "DDR5", "speed_mt_s": 5600},
                {"type": "LPDDR5X", "speed_mt_s": 8000},
            ),
        },
        "official_source_url": (
            "https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/"
            "amd-ryzen-ai-9-hx-370.html"
        ),
    },
    {
        "chip_sku": "AMD Ryzen AI 9 HX 375",
        "chip_family": "Ryzen AI 9 HX",
        "codename": "Strix Point",
        "cpu": {
            "cores": 12,
            "threads": 24,
            "max_boost_ghz": 5.1,
            "zen5_cores": 4,
            "zen5c_cores": 8,
        },
        "igpu": {
            "model": "AMD Radeon 890M",
            "cu_count": 16,
            "gfx_freq_mhz": 2900,
        },
        "npu": {
            "official_npu_tops": 55,
            "official_overall_tops": 85,
        },
        "memory": {
            "channels": 2,
            "supported_configs": (
                {"type": "DDR5", "speed_mt_s": 5600},
                {"type": "LPDDR5X", "speed_mt_s": 8000},
            ),
        },
        "official_source_url": (
            "https://www.amd.com/en/products/processors/laptop/ryzen-pro/ai-300-series/"
            "amd-ryzen-ai-9-hx-pro-375.html"
        ),
    },
    {
        "chip_sku": "AMD Ryzen AI 9 HX 470",
        "chip_family": "Ryzen AI 9 HX",
        "codename": "Gorgon Point",
        "cpu": {
            "cores": 12,
            "threads": 24,
            "max_boost_ghz": 5.2,
            "zen5_cores": 4,
            "zen5c_cores": 8,
        },
        "igpu": {
            "model": "AMD Radeon 890M",
            "cu_count": 16,
            "gfx_freq_mhz": 3100,
        },
        "npu": {
            "official_npu_tops": 55,
            "official_overall_tops": 86,
        },
        "memory": {
            "channels": 2,
            "supported_configs": (
                {"type": "DDR5", "speed_mt_s": 5600},
                {"type": "LPDDR5X", "speed_mt_s": 8533},
            ),
        },
        "official_source_url": (
            "https://www.amd.com/en/products/processors/laptop/ryzen/ai-400-series/"
            "amd-ryzen-ai-9-hx-470.html"
        ),
    },
)

SKU_MODEL_NAME_RE = re.compile(r"AMD Ryzen AI 9 HX(?: PRO)? (?P<sku>370|375|470)", re.I)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fingerprint the benchmark host and write a machine-readable hardware "
            "profile artifact for later peak and roofline analysis."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write the host profile JSON artifact.",
    )
    parser.add_argument(
        "--memory-override",
        type=str,
        default=None,
        help=(
            "Optional JSON file that supplies installed-memory fields such as "
            "type, speed_mt_s, and interface_bits when firmware tools are incomplete."
        ),
    )
    return parser.parse_args()


def _extract_json_payload(text: str) -> str:
    json_start = text.find("{")
    if json_start < 0:
        raise ValueError("Expected JSON payload in command output")
    return text[json_start:]


def _parse_colon_table(text: str) -> dict[str, str]:
    parsed = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_int(value):
    if value in ("", None):
        return None
    try:
        return int(str(value), 0)
    except ValueError:
        return None


def _parse_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_optional_text(path: Path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except PermissionError:
        return None


def parse_lscpu_output(text: str) -> dict:
    parsed = _parse_colon_table(text)
    flags = set(parsed.get("Flags", "").split())
    cpu_max_mhz = _parse_float(parsed.get("CPU max MHz"))
    return {
        "vendor_id": parsed.get("Vendor ID"),
        "model_name": parsed.get("Model name"),
        "family": _parse_int(parsed.get("CPU family")),
        "model": _parse_int(parsed.get("Model")),
        "cores": _parse_int(parsed.get("Core(s) per socket")),
        "threads": _parse_int(parsed.get("CPU(s)")),
        "threads_per_core": _parse_int(parsed.get("Thread(s) per core")),
        "max_boost_ghz": (cpu_max_mhz / 1000.0) if cpu_max_mhz is not None else None,
        "supports_avx512": any(flag.startswith("avx512") for flag in flags),
        "supports_avx512_bf16": "avx512_bf16" in flags,
        "flags": sorted(flags),
    }


def parse_cpuinfo_output(text: str) -> dict:
    first_block = text.strip().split("\n\n", 1)[0]
    parsed = _parse_colon_table(first_block)
    flags = set(parsed.get("flags", "").split())
    return {
        "model_name": parsed.get("model name"),
        "flags": sorted(flags),
        "supports_avx512": any(flag.startswith("avx512") for flag in flags),
        "supports_avx512_bf16": "avx512_bf16" in flags,
    }


def parse_rocm_smi_product_output(text: str) -> dict:
    payload = json.loads(_extract_json_payload(text))
    if not payload:
        return {}
    card = payload[next(iter(payload))]
    return {
        "detected_model": card.get("Card Series"),
        "detected_device_id": card.get("Card Model"),
        "detected_vendor": card.get("Card Vendor"),
        "detected_card_sku": card.get("Card SKU"),
        "detected_gfx_version": card.get("GFX Version"),
        "detected_subsystem_id": card.get("Subsystem ID"),
        "detected_device_rev": card.get("Device Rev"),
    }


def parse_xrt_smi_examine_json(text: str) -> dict:
    payload = json.loads(_extract_json_payload(text))
    host = payload.get("system", {}).get("host", {})
    xrt = host.get("xrt", {})
    devices = host.get("devices", [])
    device = devices[0] if devices else {}
    driver_versions = {
        driver.get("name"): driver.get("version") for driver in xrt.get("drivers", [])
    }
    return {
        "device_name": device.get("name"),
        "device_bdf": device.get("bdf"),
        "firmware_version": device.get("firmware_version"),
        "xrt_version": xrt.get("version"),
        "amdxdna_driver_version": driver_versions.get("amdxdna"),
        "virtio_pci_driver_version": driver_versions.get("virtio-pci"),
        "is_ready": device.get("is_ready"),
    }


def parse_lshw_memory_json(text: str) -> dict:
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = [payload]
    memory_node = next(
        (
            node
            for node in payload
            if node.get("class") == "memory"
            and node.get("description") == "System memory"
        ),
        {},
    )
    return {
        "installed_bytes": memory_node.get("size"),
        "type": None,
        "speed_mt_s": None,
        "interface_bits": None,
        "source": "detected" if memory_node.get("size") is not None else "unknown",
    }


def derive_theoretical_ddr_peak_bytes_per_sec(
    speed_mt_s: int, interface_bits: int
) -> float:
    return float(speed_mt_s) * 1_000_000.0 * float(interface_bits) / 8.0


def derive_igpu_fp32_peak_ops_per_sec(cu_count: int, gfx_freq_mhz: int) -> float:
    lanes_per_cu = 64.0
    fp32_ops_per_lane_per_cycle = 2.0
    return (
        float(cu_count)
        * lanes_per_cu
        * fp32_ops_per_lane_per_cycle
        * float(gfx_freq_mhz)
        * 1_000_000.0
    )


def resolve_sku_candidates(
    cpu_model_name: str | None, cpu_cores: int | None, cpu_threads: int | None
):
    exact_match = None
    if cpu_model_name:
        match = SKU_MODEL_NAME_RE.search(cpu_model_name)
        if match is not None:
            exact_match = f"AMD Ryzen AI 9 HX {match.group('sku')}"

    if exact_match is not None:
        matched = next(
            candidate
            for candidate in AMD_HX_AI_9_SKU_CANDIDATES
            if candidate["chip_sku"] == exact_match
        )
        return {
            "chip_sku": matched["chip_sku"],
            "chip_family": matched["chip_family"],
            "codename": matched["codename"],
            "match_kind": "exact_model_name",
            "candidate_matches": [matched["chip_sku"]],
            "matched_candidate": matched,
        }

    candidates = list(AMD_HX_AI_9_SKU_CANDIDATES)
    if cpu_cores is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["cpu"]["cores"] == cpu_cores
        ]
    if cpu_threads is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["cpu"]["threads"] == cpu_threads
        ]
    if len(candidates) == 1:
        matched = candidates[0]
        return {
            "chip_sku": matched["chip_sku"],
            "chip_family": matched["chip_family"],
            "codename": matched["codename"],
            "match_kind": "candidate_signature",
            "candidate_matches": [matched["chip_sku"]],
            "matched_candidate": matched,
        }
    return {
        "chip_sku": "unknown",
        "chip_family": "Ryzen AI 9 HX" if candidates else "unknown",
        "codename": "unknown",
        "match_kind": SKU_MATCH_UNKNOWN,
        "candidate_matches": [candidate["chip_sku"] for candidate in candidates],
        "matched_candidate": None,
    }


def apply_memory_override(memory: dict, override: dict | None) -> dict:
    if not override:
        return memory
    override_memory = dict(override.get("memory", override))
    merged = dict(memory)
    for key in (
        "installed_bytes",
        "type",
        "speed_mt_s",
        "interface_bits",
        "channels",
        "source",
    ):
        if key in override_memory:
            merged[key] = override_memory[key]
    if "source" not in override_memory:
        merged["source"] = "manual"
    return merged


def build_host_profile(
    *,
    lscpu_text: str,
    cpuinfo_text: str,
    rocm_smi_text: str | None = None,
    xrt_smi_json_text: str | None = None,
    lshw_memory_text: str | None = None,
    dmi_info: dict | None = None,
    memory_override: dict | None = None,
) -> dict:
    lscpu_info = parse_lscpu_output(lscpu_text)
    cpuinfo_info = parse_cpuinfo_output(cpuinfo_text)
    sku_resolution = resolve_sku_candidates(
        cpuinfo_info.get("model_name") or lscpu_info.get("model_name"),
        lscpu_info.get("cores"),
        lscpu_info.get("threads"),
    )
    matched_candidate = sku_resolution["matched_candidate"]

    cpu = dict(lscpu_info)
    cpu["model_name"] = cpuinfo_info.get("model_name") or cpu.get("model_name")
    cpu["supports_avx512"] = cpuinfo_info["supports_avx512"] or cpu["supports_avx512"]
    cpu["supports_avx512_bf16"] = (
        cpuinfo_info["supports_avx512_bf16"] or cpu["supports_avx512_bf16"]
    )
    cpu["flags"] = cpuinfo_info["flags"] or cpu["flags"]

    igpu = parse_rocm_smi_product_output(rocm_smi_text) if rocm_smi_text else {}
    if matched_candidate is not None:
        igpu.setdefault("model", matched_candidate["igpu"]["model"])
        igpu.setdefault("cu_count", matched_candidate["igpu"]["cu_count"])
        igpu.setdefault("gfx_freq_mhz", matched_candidate["igpu"]["gfx_freq_mhz"])

    npu = parse_xrt_smi_examine_json(xrt_smi_json_text) if xrt_smi_json_text else {}
    if matched_candidate is not None:
        npu.setdefault(
            "official_npu_tops", matched_candidate["npu"]["official_npu_tops"]
        )
        npu.setdefault(
            "official_overall_tops", matched_candidate["npu"]["official_overall_tops"]
        )

    memory = (
        parse_lshw_memory_json(lshw_memory_text)
        if lshw_memory_text
        else {
            "installed_bytes": None,
            "type": None,
            "speed_mt_s": None,
            "interface_bits": None,
            "source": "unknown",
        }
    )
    memory = apply_memory_override(memory, memory_override)
    if matched_candidate is not None:
        memory.setdefault("channels", matched_candidate["memory"]["channels"])
        memory.setdefault(
            "supported_configs", list(matched_candidate["memory"]["supported_configs"])
        )
    if memory.get("speed_mt_s") and memory.get("interface_bits"):
        memory["derived_theoretical_peak_bytes_per_sec"] = (
            derive_theoretical_ddr_peak_bytes_per_sec(
                int(memory["speed_mt_s"]), int(memory["interface_bits"])
            )
        )

    return {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chip_sku": sku_resolution["chip_sku"],
        "chip_family": sku_resolution["chip_family"],
        "chip_codename": sku_resolution["codename"],
        "sku_match_kind": sku_resolution["match_kind"],
        "sku_candidates_considered": sku_resolution["candidate_matches"],
        "cpu": cpu,
        "igpu": igpu,
        "npu": npu,
        "memory": memory,
        "system": dict(dmi_info or {}),
        "matched_reference": (
            {
                "chip_sku": matched_candidate["chip_sku"],
                "official_source_url": matched_candidate["official_source_url"],
            }
            if matched_candidate is not None
            else None
        ),
    }


def _run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise RuntimeError(f"{' '.join(command)}: {message}")
    return result.stdout


def collect_xrt_smi_examine_json():
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "xrt_examine.json"
        _run_command(["xrt-smi", "examine", "-f", "JSON", "-o", str(output_path)])
        return output_path.read_text(encoding="utf-8")


def collect_dmi_info():
    dmi_root = Path("/sys/class/dmi/id")
    return {
        "board_name": _read_optional_text(dmi_root / "board_name"),
        "product_name": _read_optional_text(dmi_root / "product_name"),
        "product_version": _read_optional_text(dmi_root / "product_version"),
        "bios_version": _read_optional_text(dmi_root / "bios_version"),
    }


def load_memory_override(path: str | None):
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_host_profile(memory_override_path: str | None = None):
    rocm_smi_text = None
    xrt_smi_json_text = None
    lshw_memory_text = None

    try:
        rocm_smi_text = _run_command(["rocm-smi", "--showproductname", "--json"])
    except (FileNotFoundError, RuntimeError):
        rocm_smi_text = None

    try:
        xrt_smi_json_text = collect_xrt_smi_examine_json()
    except (FileNotFoundError, RuntimeError):
        xrt_smi_json_text = None

    try:
        lshw_memory_text = _run_command(["lshw", "-json", "-class", "memory"])
    except (FileNotFoundError, RuntimeError):
        lshw_memory_text = None

    return build_host_profile(
        lscpu_text=_run_command(["lscpu"]),
        cpuinfo_text=Path("/proc/cpuinfo").read_text(encoding="utf-8"),
        rocm_smi_text=rocm_smi_text,
        xrt_smi_json_text=xrt_smi_json_text,
        lshw_memory_text=lshw_memory_text,
        dmi_info=collect_dmi_info(),
        memory_override=load_memory_override(memory_override_path),
    )


def _measured_peak_entry(
    measured_calibration: dict | None, backend: str, dtype: str
) -> dict:
    return (
        measured_calibration.get("peaks", {}).get(backend, {}).get(dtype, {})
        if measured_calibration
        else {}
    )


def _row_tool_versions(host_profile: dict, measured_calibration: dict | None) -> dict:
    tool_versions = dict((measured_calibration or {}).get("tool_versions", {}))
    xrt_version = host_profile.get("npu", {}).get("xrt_version")
    if xrt_version is not None:
        tool_versions.setdefault("xrt_smi", xrt_version)
    amdxdna_driver_version = host_profile.get("npu", {}).get("amdxdna_driver_version")
    if amdxdna_driver_version is not None:
        tool_versions.setdefault("amdxdna", amdxdna_driver_version)
    return tool_versions


def _build_peak_reference_row(
    *,
    host_profile: dict,
    measured_calibration: dict | None,
    backend: str,
    dtype: str,
    official_peak_ops_per_sec: float | None = None,
    derived_theoretical_peak_ops_per_sec: float | None = None,
    measured_entry: dict | None = None,
    primary_preference: tuple[str, ...] = ("measured", "derived", "official"),
    peak_source_note: str = "",
) -> dict:
    measured_entry = measured_entry or {}
    measured_peak_ops_per_sec = measured_entry.get("measured_peak_ops_per_sec")
    source_values = {
        "official": official_peak_ops_per_sec,
        "derived": derived_theoretical_peak_ops_per_sec,
        "measured": measured_peak_ops_per_sec,
    }
    primary_peak_source_kind = "unknown"
    primary_peak_ops_per_sec = None
    for source_kind in primary_preference:
        source_value = source_values.get(source_kind)
        if source_value is not None:
            primary_peak_source_kind = source_kind
            primary_peak_ops_per_sec = source_value
            break

    notes = []
    if peak_source_note:
        notes.append(peak_source_note)
    measured_note = measured_entry.get("note")
    if measured_note:
        notes.append(measured_note)

    return {
        "backend": backend,
        "dtype": dtype,
        "chip_sku": host_profile.get("chip_sku"),
        "chip_family": host_profile.get("chip_family"),
        "chip_codename": host_profile.get("chip_codename"),
        "official_peak_ops_per_sec": official_peak_ops_per_sec,
        "derived_theoretical_peak_ops_per_sec": derived_theoretical_peak_ops_per_sec,
        "measured_peak_ops_per_sec": measured_peak_ops_per_sec,
        "primary_peak_ops_per_sec": primary_peak_ops_per_sec,
        "primary_peak_source_kind": primary_peak_source_kind,
        "peak_source_note": " ".join(notes).strip(),
        "tool_versions": _row_tool_versions(host_profile, measured_calibration),
    }


def build_peak_reference_artifact(
    host_profile: dict, measured_calibration: dict | None = None
):
    measured_calibration = measured_calibration or {}
    references = []

    references.append(
        _build_peak_reference_row(
            host_profile=host_profile,
            measured_calibration=measured_calibration,
            backend="cpu",
            dtype="bfloat16",
            measured_entry=_measured_peak_entry(
                measured_calibration, "cpu", "bfloat16"
            ),
            primary_preference=("measured",),
            peak_source_note=(
                "CPU bf16 percent-of-peak should use a measured GEMM calibration peak."
            ),
        )
    )
    references.append(
        _build_peak_reference_row(
            host_profile=host_profile,
            measured_calibration=measured_calibration,
            backend="cpu",
            dtype="float32",
            measured_entry=_measured_peak_entry(measured_calibration, "cpu", "float32"),
            primary_preference=("measured",),
            peak_source_note=(
                "CPU fp32 percent-of-peak should use a measured GEMM calibration peak."
            ),
        )
    )

    igpu = host_profile.get("igpu", {})
    derived_igpu_fp32_peak = None
    if igpu.get("cu_count") is not None and igpu.get("gfx_freq_mhz") is not None:
        derived_igpu_fp32_peak = derive_igpu_fp32_peak_ops_per_sec(
            int(igpu["cu_count"]), int(igpu["gfx_freq_mhz"])
        )
    references.append(
        _build_peak_reference_row(
            host_profile=host_profile,
            measured_calibration=measured_calibration,
            backend="igpu",
            dtype="bfloat16",
            measured_entry=_measured_peak_entry(
                measured_calibration, "igpu", "bfloat16"
            ),
            primary_preference=("measured",),
            peak_source_note=(
                "iGPU bf16 percent-of-peak should use a measured GEMM calibration peak."
            ),
        )
    )
    references.append(
        _build_peak_reference_row(
            host_profile=host_profile,
            measured_calibration=measured_calibration,
            backend="igpu",
            dtype="float16",
            measured_entry=_measured_peak_entry(
                measured_calibration, "igpu", "float16"
            ),
            primary_preference=("measured",),
            peak_source_note=(
                "iGPU fp16 percent-of-peak should use a measured GEMM calibration peak."
            ),
        )
    )
    references.append(
        _build_peak_reference_row(
            host_profile=host_profile,
            measured_calibration=measured_calibration,
            backend="igpu",
            dtype="float32",
            derived_theoretical_peak_ops_per_sec=derived_igpu_fp32_peak,
            measured_entry=_measured_peak_entry(
                measured_calibration, "igpu", "float32"
            ),
            primary_preference=("measured", "derived"),
            peak_source_note=(
                "Derived iGPU fp32 peak is a secondary sanity check until a measured fp32 GEMM "
                "calibration is captured."
            ),
        )
    )

    official_npu_tops = host_profile.get("npu", {}).get("official_npu_tops")
    official_npu_ops_per_sec = (
        float(official_npu_tops) * 1_000_000_000_000.0
        if official_npu_tops is not None
        else None
    )
    references.append(
        _build_peak_reference_row(
            host_profile=host_profile,
            measured_calibration=measured_calibration,
            backend="npu",
            dtype="int8",
            official_peak_ops_per_sec=official_npu_ops_per_sec,
            measured_entry=_measured_peak_entry(measured_calibration, "npu", "int8"),
            primary_preference=("measured", "official"),
            peak_source_note=(
                "Vendor NPU TOPS and xrt-smi validate INT8 GEMM are reference metadata, not the "
                "bf16 primary denominator."
            ),
        )
    )
    references.append(
        _build_peak_reference_row(
            host_profile=host_profile,
            measured_calibration=measured_calibration,
            backend="npu",
            dtype="bfloat16",
            measured_entry=_measured_peak_entry(
                measured_calibration, "npu", "bfloat16"
            ),
            primary_preference=("measured",),
            peak_source_note=(
                "NPU bf16 percent-of-peak should use a measured IRON calibration peak."
            ),
        )
    )

    memory = host_profile.get("memory", {})
    measured_bandwidth = measured_calibration.get("memory_bandwidth", {})
    theoretical_peak_bytes_per_sec = memory.get(
        "derived_theoretical_peak_bytes_per_sec"
    )
    measured_peak_bytes_per_sec = measured_bandwidth.get("measured_peak_bytes_per_sec")
    primary_peak_bytes_per_sec = measured_peak_bytes_per_sec
    primary_peak_source_kind = (
        "measured" if measured_peak_bytes_per_sec is not None else "unknown"
    )
    if (
        primary_peak_bytes_per_sec is None
        and theoretical_peak_bytes_per_sec is not None
    ):
        primary_peak_bytes_per_sec = theoretical_peak_bytes_per_sec
        primary_peak_source_kind = "derived"

    return {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chip_sku": host_profile.get("chip_sku"),
        "chip_family": host_profile.get("chip_family"),
        "chip_codename": host_profile.get("chip_codename"),
        "host_profile_artifact_version": host_profile.get("artifact_version"),
        "tool_versions": _row_tool_versions(host_profile, measured_calibration),
        "references": references,
        "memory_bandwidth": {
            "chip_sku": host_profile.get("chip_sku"),
            "chip_family": host_profile.get("chip_family"),
            "chip_codename": host_profile.get("chip_codename"),
            "theoretical_peak_bytes_per_sec": theoretical_peak_bytes_per_sec,
            "measured_copy_bytes_per_sec": measured_bandwidth.get("copy_bytes_per_sec"),
            "measured_triad_bytes_per_sec": measured_bandwidth.get(
                "triad_bytes_per_sec"
            ),
            "measured_peak_bytes_per_sec": measured_peak_bytes_per_sec,
            "primary_peak_bytes_per_sec": primary_peak_bytes_per_sec,
            "primary_peak_source_kind": primary_peak_source_kind,
            "peak_source_note": measured_bandwidth.get(
                "note",
                "STREAM-style sustained bandwidth should be the primary roofline memory ceiling.",
            ),
            "tool_versions": _row_tool_versions(host_profile, measured_calibration),
        },
    }


def main():
    args = parse_args()
    profile = collect_host_profile(args.memory_override)
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote host profile to {output_path}")


if __name__ == "__main__":
    main()
