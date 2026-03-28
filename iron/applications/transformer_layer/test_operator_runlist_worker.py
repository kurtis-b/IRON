# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.applications.transformer_layer.operator_runlist_worker import (
    benchmark_operator_runlist_request,
    parity_operator_runlist_request,
)
from iron.applications.transformer_layer.src.pipeline.operator_runlist_worker import (
    benchmark_operator_runlist_request as structured_benchmark_operator_runlist_request,
)
from iron.applications.transformer_layer.src.pipeline.operator_runlist_worker import (
    parity_operator_runlist_request as structured_parity_operator_runlist_request,
)
from iron.applications.transformer_layer.src.pipeline.operator_runlist_worker import (
    run_worker,
)


def test_operator_runlist_worker_restructure_preserves_legacy_imports_and_dispatch():
    assert (
        benchmark_operator_runlist_request
        is structured_benchmark_operator_runlist_request
    )
    assert parity_operator_runlist_request is structured_parity_operator_runlist_request

    with pytest.raises(ValueError, match="Unsupported runlist worker mode"):
        run_worker({"mode": "invalid"})
