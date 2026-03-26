#!/usr/bin/env bash
set -euo pipefail

echo "1. Verify synthetic reference-layer tests pass."
echo "2. Run transformer_layer/npu_inference.py for one seq_len and one execution_mode."
echo "3. Validate parity on a small synthetic case."
