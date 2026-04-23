#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "missing required command: ${name}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "missing required path: ${path}" >&2
    exit 1
  fi
}

require_command python3
require_command bash
require_command find

if [[ -f /opt/xilinx/xrt/setup.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/xilinx/xrt/setup.sh
fi

require_file "${ROOT_DIR}/ironenv/bin/activate"
# shellcheck disable=SC1091
source "${ROOT_DIR}/ironenv/bin/activate"

require_command xrt-smi
require_command amd-ttm

python3 - <<'PY'
import importlib
for module_name in (
    "torch",
    "numpy",
    "ml_dtypes",
    "iron.applications.transformer_layer.study.unattended_reboot",
):
    importlib.import_module(module_name)
print("python imports: ok")
PY

echo "environment check: ok"
