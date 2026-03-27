#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${TRANSFORMER_LAYER_POWER_CYCLE_ENV:-/etc/transformer-layer-power-cycle.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

run_command() {
  local command="${1:-}"
  if [[ -z "${command}" ]]; then
    echo "power-cycle command is empty" >&2
    exit 1
  fi
  /usr/bin/env bash -lc "${command}"
}

if [[ -n "${POWER_CYCLE_COMMAND:-}" ]]; then
  run_command "${POWER_CYCLE_COMMAND}"
  exit 0
fi

if [[ -z "${POWER_OFF_COMMAND:-}" || -z "${POWER_ON_COMMAND:-}" ]]; then
  echo "Define POWER_CYCLE_COMMAND or both POWER_OFF_COMMAND and POWER_ON_COMMAND" >&2
  exit 1
fi

run_command "${POWER_OFF_COMMAND}"
sleep "${POWER_OFF_WAIT_SEC:-2}"
run_command "${POWER_ON_COMMAND}"
sleep "${POWER_ON_WAIT_SEC:-5}"
