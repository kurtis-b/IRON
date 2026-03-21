#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ACTIVE_POWER_FIELDNAMES = [
    "power_backend",
    "power_sample_count",
    "power_window_sec",
    "avg_pkg_watt",
    "max_pkg_watt",
    "avg_cor_watt",
    "max_cor_watt",
    "avg_gfx_watt",
    "max_gfx_watt",
    "avg_ram_watt",
    "max_ram_watt",
    "power_log",
]

_POWER_COLUMNS = ("PkgWatt", "CorWatt", "GFXWatt", "RAMWatt")
_ROCM_SMI_PREFERRED_POWER_KEYS = (
    "Current Socket Graphics Package Power (W)",
    "Average Socket Graphics Package Power (W)",
    "Average Graphics Package Power (W)",
    "Current Graphics Package Power (W)",
)
_POWERCAP_ROOT = Path("/sys/class/powercap")
_POWERCAP_HELPER = Path(__file__).with_name("read_powercap_rapl.py")
_USER_POWER_BACKEND_CHOICES = ("none", "auto")


def add_power_measurement_args(parser, default_backend="none"):
    parser.add_argument(
        "--power-backend",
        choices=_USER_POWER_BACKEND_CHOICES,
        default=default_backend,
        help=(
            "Timed-region power measurement backend. auto uses powercap-rapl for cpu, "
            "turbostat for npu, and rocm-smi for igpu."
        ),
    )
    parser.add_argument(
        "--power-interval-sec",
        type=float,
        default=0.5,
        help="Polling interval used by timed-region power logging.",
    )
    parser.add_argument(
        "--power-log-path",
        type=str,
        default=None,
        help="Optional power log file path for the current benchmark segment.",
    )


def resolve_power_backend(power_backend, mode):
    if power_backend == "none":
        return "none"
    if power_backend != "auto":
        raise ValueError(
            f"Unsupported power_backend={power_backend!r}; expected one of "
            f"{_USER_POWER_BACKEND_CHOICES}"
        )
    if mode == "cpu":
        return "powercap-rapl"
    if mode == "igpu":
        return "rocm-smi"
    return "turbostat"


def empty_power_stats(
    sample_count_key="power_sample_count", window_key="power_window_sec"
):
    return {
        sample_count_key: 0,
        window_key: "",
        "avg_pkg_watt": "",
        "max_pkg_watt": "",
        "avg_cor_watt": "",
        "max_cor_watt": "",
        "avg_gfx_watt": "",
        "max_gfx_watt": "",
        "avg_ram_watt": "",
        "max_ram_watt": "",
    }


def _extract_power_value(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(raw_value))
    if match is None:
        return None
    return float(match.group(0))


def read_rocm_smi_power(device_index):
    result = subprocess.run(
        ["rocm-smi", "--showpower", "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "rocm-smi power query failed\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse rocm-smi JSON power output: {exc}\n"
            f"STDOUT:\n{result.stdout}"
        ) from exc

    card_key = f"card{device_index}"
    card_data = payload.get(card_key)
    if card_data is None and device_index == 0:
        card_entries = [
            (key, value)
            for key, value in payload.items()
            if re.fullmatch(r"card\d+", key)
        ]
        if len(card_entries) == 1:
            _, card_data = card_entries[0]
    if not isinstance(card_data, dict):
        raise RuntimeError(
            f"rocm-smi JSON output did not include power data for device index "
            f"{device_index}: keys={sorted(payload.keys())}"
        )

    for key in _ROCM_SMI_PREFERRED_POWER_KEYS:
        if key not in card_data:
            continue
        power_value = _extract_power_value(card_data[key])
        if power_value is not None:
            return power_value

    for key, raw_value in card_data.items():
        if "power" not in key.lower():
            continue
        power_value = _extract_power_value(raw_value)
        if power_value is not None:
            return power_value
    raise RuntimeError(
        f"rocm-smi JSON output did not include a usable power reading for {card_key}: "
        f"{card_data}"
    )


def _is_power_header(tokens):
    return bool(tokens) and all(token in _POWER_COLUMNS for token in tokens)


def summarize_power_samples(samples, duration_sec):
    stats = empty_power_stats()
    stats["power_sample_count"] = len(samples)
    stats["power_window_sec"] = (
        f"{duration_sec:.6f}" if duration_sec is not None else ""
    )
    if not samples:
        return stats

    def summarize(column, avg_key, max_key):
        values = [
            sample[column] for sample in samples if sample.get(column) is not None
        ]
        if not values:
            return
        stats[avg_key] = f"{sum(values) / len(values):.6f}"
        stats[max_key] = f"{max(values):.6f}"

    summarize("PkgWatt", "avg_pkg_watt", "max_pkg_watt")
    summarize("CorWatt", "avg_cor_watt", "max_cor_watt")
    summarize("GFXWatt", "avg_gfx_watt", "max_gfx_watt")
    summarize("RAMWatt", "avg_ram_watt", "max_ram_watt")
    return stats


def _iter_powercap_zone_dirs(powercap_root):
    root = Path(powercap_root)
    for entry in sorted(root.iterdir()):
        if (entry / "name").exists():
            yield entry


def discover_powercap_rapl_zones(powercap_root=_POWERCAP_ROOT):
    root = Path(powercap_root)
    package_zones = []
    core_zones = []

    for zone_dir in sorted(_iter_powercap_zone_dirs(root)):
        name_path = zone_dir / "name"
        energy_path = zone_dir / "energy_uj"
        max_energy_range_path = zone_dir / "max_energy_range_uj"
        if not energy_path.exists() or not max_energy_range_path.exists():
            continue
        zone_name = name_path.read_text(encoding="utf-8").strip()
        zone = {
            "name": zone_name,
            "energy_path": str(energy_path),
            "max_energy_range_uj": int(
                max_energy_range_path.read_text(encoding="utf-8").strip()
            ),
        }
        if zone_name.startswith("package-"):
            package_zones.append(zone)
        elif zone_name == "core":
            core_zones.append(zone)

    if not package_zones:
        raise RuntimeError(
            f"powercap-rapl did not expose any package energy zones under {root}"
        )

    return {
        "package": package_zones,
        "core": core_zones,
    }


def _read_powercap_energy_values_direct(energy_paths):
    values = {}
    for energy_path in energy_paths:
        values[str(energy_path)] = int(
            Path(energy_path).read_text(encoding="utf-8").strip()
        )
    return values


def _read_powercap_energy_values_via_helper(
    energy_paths,
    helper_path=_POWERCAP_HELPER,
):
    helper = Path(helper_path)
    if not helper.exists():
        raise RuntimeError(f"Missing powercap helper script: {helper}")
    result = subprocess.run(
        ["sudo", "-n", str(helper), *[str(path) for path in energy_paths]],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "powercap-rapl helper failed\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse powercap helper JSON output: {exc}\n"
            f"STDOUT:\n{result.stdout}"
        ) from exc
    return {str(path): int(payload[str(path)]) for path in energy_paths}


def read_powercap_energy_values(energy_paths, helper_path=_POWERCAP_HELPER):
    try:
        return _read_powercap_energy_values_direct(energy_paths)
    except PermissionError as direct_exc:
        try:
            return _read_powercap_energy_values_via_helper(
                energy_paths,
                helper_path=helper_path,
            )
        except Exception as helper_exc:
            raise RuntimeError(
                "powercap-rapl energy counters are not readable. Grant direct read "
                f"access to {Path(energy_paths[0]).parent.parent} energy_uj files or "
                f"allow sudo for {Path(helper_path)}"
            ) from helper_exc


def read_powercap_rapl_snapshot(zones, helper_path=_POWERCAP_HELPER):
    energy_paths = [
        zone["energy_path"]
        for group_name in ("package", "core")
        for zone in zones.get(group_name, [])
    ]
    if not energy_paths:
        return {}
    return read_powercap_energy_values(energy_paths, helper_path=helper_path)


def powercap_energy_delta_uj(start_value, end_value, max_energy_range_uj):
    delta = end_value - start_value
    if delta < 0:
        delta += max_energy_range_uj
    return max(0, delta)


def summarize_powercap_rapl(
    zones,
    start_snapshot,
    end_snapshot,
    duration_sec,
):
    stats = empty_power_stats()
    if duration_sec is None or duration_sec <= 0:
        return stats, []

    zone_rows = []
    package_delta_uj = 0
    core_delta_uj = 0
    for group_name in ("package", "core"):
        for zone in zones.get(group_name, []):
            energy_path = zone["energy_path"]
            start_value = start_snapshot[energy_path]
            end_value = end_snapshot[energy_path]
            delta_uj = powercap_energy_delta_uj(
                start_value,
                end_value,
                zone["max_energy_range_uj"],
            )
            avg_watt = delta_uj / duration_sec / 1e6
            zone_rows.append(
                {
                    "group": group_name,
                    "name": zone["name"],
                    "energy_path": energy_path,
                    "start_uj": start_value,
                    "end_uj": end_value,
                    "delta_uj": delta_uj,
                    "avg_watt": avg_watt,
                }
            )
            if group_name == "package":
                package_delta_uj += delta_uj
            else:
                core_delta_uj += delta_uj

    stats["power_sample_count"] = 2
    stats["power_window_sec"] = f"{duration_sec:.6f}"
    if package_delta_uj > 0:
        stats["avg_pkg_watt"] = f"{package_delta_uj / duration_sec / 1e6:.6f}"
    if core_delta_uj > 0:
        stats["avg_cor_watt"] = f"{core_delta_uj / duration_sec / 1e6:.6f}"
    return stats, zone_rows


def _write_power_log(log_path, duration_sec, header, samples):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{duration_sec:.6f} sec\n")
        f.write("\t".join(header) + "\n")
        for sample in samples:
            row = []
            for name in header:
                value = sample.get(name)
                row.append("" if value is None else f"{value:.6f}")
            f.write("\t".join(row) + "\n")


def _write_powercap_rapl_log(log_path, duration_sec, zone_rows):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{duration_sec:.6f} sec\n")
        f.write("Group\tZone\tStart_uJ\tEnd_uJ\tDelta_uJ\tAvgWatt\tEnergyPath\n")
        for row in zone_rows:
            f.write(
                "\t".join(
                    [
                        row["group"],
                        row["name"],
                        str(row["start_uj"]),
                        str(row["end_uj"]),
                        str(row["delta_uj"]),
                        f"{row['avg_watt']:.6f}",
                        row["energy_path"],
                    ]
                )
                + "\n"
            )


def parse_power_log(log_path, duration_sec=None):
    path = Path(log_path)
    if not path.exists():
        raise RuntimeError(f"Missing power log: {log_path}")

    parsed_duration_sec = None
    header = None
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if parsed_duration_sec is None and line.endswith(" sec"):
                try:
                    parsed_duration_sec = float(line.split()[0])
                except (IndexError, ValueError):
                    parsed_duration_sec = None
                continue

            tokens = line.split()
            if _is_power_header(tokens):
                header = tokens
                continue
            if header is None or len(tokens) != len(header):
                continue

            sample = {}
            all_zero = True
            for name, value in zip(header, tokens):
                try:
                    sample[name] = float(value)
                    if sample[name] != 0.0:
                        all_zero = False
                except ValueError:
                    sample[name] = None
                    all_zero = False
            if all_zero:
                continue
            samples.append(sample)
    return summarize_power_samples(
        samples,
        duration_sec if duration_sec is not None else parsed_duration_sec,
    )


def format_power_stats_for_csv(power_backend, power_stats, power_log_path):
    return {
        "power_backend": power_backend,
        "power_sample_count": str(power_stats["power_sample_count"]),
        "power_window_sec": power_stats["power_window_sec"],
        "avg_pkg_watt": power_stats["avg_pkg_watt"],
        "max_pkg_watt": power_stats["max_pkg_watt"],
        "avg_cor_watt": power_stats["avg_cor_watt"],
        "max_cor_watt": power_stats["max_cor_watt"],
        "avg_gfx_watt": power_stats["avg_gfx_watt"],
        "max_gfx_watt": power_stats["max_gfx_watt"],
        "avg_ram_watt": power_stats["avg_ram_watt"],
        "max_ram_watt": power_stats["max_ram_watt"],
        "power_log": str(power_log_path) if power_log_path is not None else "",
    }


def power_stats_from_row(row):
    return {
        "power_sample_count": int(row.get("power_sample_count", "0") or 0),
        "power_window_sec": row.get("power_window_sec", ""),
        "avg_pkg_watt": row.get("avg_pkg_watt", ""),
        "max_pkg_watt": row.get("max_pkg_watt", ""),
        "avg_cor_watt": row.get("avg_cor_watt", ""),
        "max_cor_watt": row.get("max_cor_watt", ""),
        "avg_gfx_watt": row.get("avg_gfx_watt", ""),
        "max_gfx_watt": row.get("max_gfx_watt", ""),
        "avg_ram_watt": row.get("avg_ram_watt", ""),
        "max_ram_watt": row.get("max_ram_watt", ""),
    }


def effective_device_watt(power_stats, mode, prefix="avg"):
    preferred = (
        power_stats.get(f"{prefix}_gfx_watt", "")
        if mode == "igpu"
        else power_stats.get(f"{prefix}_pkg_watt", "")
    )
    fallback = power_stats.get(f"{prefix}_pkg_watt", "")
    if preferred not in ("", None):
        return preferred
    return fallback


def derive_power_log_path(base_path, *suffix_parts):
    if base_path is None:
        return None
    path = Path(base_path)
    tags = [str(part) for part in suffix_parts if part not in (None, "")]
    if not tags:
        return path
    return path.with_name(f"{path.stem}_{'_'.join(tags)}{path.suffix}")


class _NullPowerMonitor:
    def __init__(self):
        self.stats = empty_power_stats()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _RocmSmiPowerMonitor:
    def __init__(self, interval_sec, device_index, log_path):
        self._interval_sec = max(interval_sec, 0.05)
        self._device_index = device_index
        self._log_path = Path(log_path) if log_path is not None else None
        self._samples = []
        self._error = None
        self._stop_event = threading.Event()
        self._thread = None
        self._start = None
        self.stats = empty_power_stats()

    def __enter__(self):
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                raise RuntimeError("Timed rocm-smi power sampling thread did not stop")
        if self._error is not None:
            raise self._error
        duration_sec = time.perf_counter() - self._start
        sample_rows = [{"GFXWatt": sample} for sample in self._samples]
        self.stats = summarize_power_samples(sample_rows, duration_sec)
        if self._log_path is not None:
            _write_power_log(self._log_path, duration_sec, ["GFXWatt"], sample_rows)
        return False

    def _run(self):
        next_sample_at = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now >= next_sample_at:
                try:
                    self._samples.append(read_rocm_smi_power(self._device_index))
                except Exception as exc:
                    self._error = exc
                    self._stop_event.set()
                    return
                next_sample_at = now + self._interval_sec
            if self._stop_event.wait(max(0.0, next_sample_at - time.perf_counter())):
                return


class _PowercapRaplPowerMonitor:
    def __init__(
        self, log_path, powercap_root=_POWERCAP_ROOT, helper_path=_POWERCAP_HELPER
    ):
        self._log_path = Path(log_path) if log_path is not None else None
        self._powercap_root = Path(powercap_root)
        self._helper_path = Path(helper_path)
        self._zones = None
        self._start = None
        self._start_snapshot = None
        self.stats = empty_power_stats()

    def __enter__(self):
        self._zones = discover_powercap_rapl_zones(self._powercap_root)
        self._start_snapshot = read_powercap_rapl_snapshot(
            self._zones,
            helper_path=self._helper_path,
        )
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration_sec = time.perf_counter() - self._start
        end_snapshot = read_powercap_rapl_snapshot(
            self._zones,
            helper_path=self._helper_path,
        )
        self.stats, zone_rows = summarize_powercap_rapl(
            self._zones,
            self._start_snapshot,
            end_snapshot,
            duration_sec,
        )
        if self._log_path is not None:
            _write_powercap_rapl_log(self._log_path, duration_sec, zone_rows)
        return False


class _TurbostatPowerMonitor:
    def __init__(self, interval_sec, log_path):
        self._interval_sec = max(interval_sec, 0.05)
        self._requested_log_path = Path(log_path) if log_path is not None else None
        self._temp_log_path = None
        self._process = None
        self._start = None
        self.stats = empty_power_stats()

    @property
    def _log_path(self):
        if self._requested_log_path is not None:
            return self._requested_log_path
        if self._temp_log_path is None:
            handle = tempfile.NamedTemporaryFile(
                prefix="bert_power_", suffix=".log", delete=False
            )
            handle.close()
            self._temp_log_path = Path(handle.name)
        return self._temp_log_path

    def __enter__(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._start = time.perf_counter()
        self._process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "turbostat",
                "--Summary",
                "--quiet",
                "--show",
                "PkgWatt,CorWatt,GFXWatt,RAMWatt",
                "--interval",
                str(self._interval_sec),
                "--out",
                str(self._log_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._process is None:
            return False
        self._process.send_signal(signal.SIGINT)
        stdout, stderr = self._process.communicate(timeout=30.0)
        if self._process.returncode != 0:
            raise RuntimeError(
                "turbostat power sampling failed\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        duration_sec = time.perf_counter() - self._start
        self.stats = parse_power_log(self._log_path, duration_sec=duration_sec)
        if self._requested_log_path is None and self._temp_log_path is not None:
            self._temp_log_path.unlink(missing_ok=True)
        return False


def create_power_monitor(backend, interval_sec, device_index=0, log_path=None):
    if backend == "none":
        return _NullPowerMonitor()
    if backend == "powercap-rapl":
        return _PowercapRaplPowerMonitor(log_path)
    if backend == "rocm-smi":
        return _RocmSmiPowerMonitor(interval_sec, device_index, log_path)
    if backend == "turbostat":
        return _TurbostatPowerMonitor(interval_sec, log_path)
    raise ValueError(f"Unsupported power backend: {backend}")


def measure_power_for_duration(
    backend,
    duration_sec,
    interval_sec,
    *,
    device_index=0,
    log_path=None,
):
    with create_power_monitor(
        backend,
        interval_sec,
        device_index=device_index,
        log_path=log_path,
    ) as monitor:
        time.sleep(max(0.0, duration_sec))
    return monitor.stats
