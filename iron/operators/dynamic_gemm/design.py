# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

import argparse
import numpy as np
import sys

from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU2, Tile
from aie.iron.placers import SequentialPlacer

from iron.operators.gemm.design import microkernel_mac_dim_map


def main():
    argparser = argparse.ArgumentParser(
        prog="Dynamic GEMM MLIR Design",
        description="Emit MLIR code for dynamic GEMM with runtime-sequence tail handling",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2")
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument("--n-aie-cols", type=int, choices=[4, 8], default=4)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--scalar", type=bool, choices=[0, 1], default=0)
    argparser.add_argument(
        "--emulate-bf16-mmul-with-bfp16", action="store_true", default=False
    )
    argparser.add_argument("--prio-accuracy", action="store_true", default=False)
    argparser.add_argument("--separate-c-tiles", type=int, choices=[0, 1], default=1)
    argparser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Name of the archive file for the AIE kernels",
    )
    argparser.add_argument("--dtype_in", type=str, choices=["bf16"], default="bf16")
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "f32"],
        default="bf16",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns for runtime transfers",
    )
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        help="Output file path for the generated MLIR module",
    )

    args = argparser.parse_args()
    maybe_module = my_matmul(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.n_aie_cols,
        args.dtype_in,
        args.dtype_out,
        args.b_col_maj,
        args.c_col_maj,
        args.scalar,
        args.emulate_bf16_mmul_with_bfp16,
        args.prio_accuracy,
        args.separate_c_tiles,
        args.trace_size,
        args.archive,
        args.generate_taps,
    )

    if args.generate_taps:
        return maybe_module

    output_file_path = Path(args.output_file_path)
    with open(output_file_path, "w") as f:
        f.write(str(maybe_module))


def ceildiv(a, b):
    return (a + b - 1) // b


@dataclass(frozen=True)
class DynamicGemmRuntimeTask:
    tensor_name: str
    action: str
    phase: str
    group_id: object
    endpoint_index: int
    compute_tiles: tuple[tuple[int, int], ...]
    tap: TensorAccessPattern


@dataclass(frozen=True)
class DynamicGemmPhasePlan:
    name: str
    worker_tile_counts: tuple[tuple[int, ...], ...]
    tasks: tuple[DynamicGemmRuntimeTask, ...]

    @property
    def has_work(self) -> bool:
        return any(count > 0 for row in self.worker_tile_counts for count in row)


@dataclass(frozen=True)
class DynamicGemmTransferPlan:
    phases: tuple[DynamicGemmPhasePlan, ...]

    @property
    def tasks(self) -> tuple[DynamicGemmRuntimeTask, ...]:
        return tuple(task for phase in self.phases for task in phase.tasks)

    @property
    def A_taps(self) -> TensorAccessSequence:
        return TensorAccessSequence.from_taps(
            tuple(
                task.tap
                for task in self.tasks
                if task.tensor_name == "A" and task.action == "fill"
            )
        )

    @property
    def B_taps(self) -> TensorAccessSequence:
        return TensorAccessSequence.from_taps(
            tuple(
                task.tap
                for task in self.tasks
                if task.tensor_name == "B" and task.action == "fill"
            )
        )

    @property
    def C_taps(self) -> TensorAccessSequence:
        return TensorAccessSequence.from_taps(
            tuple(
                task.tap
                for task in self.tasks
                if task.tensor_name == "C" and task.action == "drain"
            )
        )


def _row_compute_tiles(row: int, n_aie_cols: int) -> tuple[tuple[int, int], ...]:
    return tuple((row, col) for col in range(n_aie_cols))


def _column_compute_tiles(
    column: int, n_aie_rows: int = 4
) -> tuple[tuple[int, int], ...]:
    return tuple((row, column) for row in range(n_aie_rows))


def _make_a_column_repeat_tap(M, K, row_offset, m, k, K_div_k, col_repeat):
    return TensorAccessPattern(
        (M, K),
        offset=row_offset * K,
        sizes=[col_repeat, K_div_k, m, k],
        strides=[0, k, K, 1],
    )


def _make_a_row_repeat_tap(
    M, K, row_offset, m, k, K_div_k, row_repeat, row_block_stride
):
    return TensorAccessPattern(
        (M, K),
        offset=row_offset * K,
        sizes=[row_repeat, K_div_k, m, k],
        strides=[row_block_stride * K, k, K, 1],
    )


def _make_a_row_repeat_sink_tap(
    M,
    K,
    n,
    row_repeat,
    row_block_height,
    sink_column_index,
):
    return TensorAccessPattern(
        (M, K),
        offset=sink_column_index * n,
        sizes=[row_repeat, 1, row_block_height, n],
        strides=[row_block_height * K, 0, K, 1],
    )


def _make_b_column_block_tap(
    K,
    N,
    k,
    n,
    K_div_k,
    mem_tile_n,
    b_col_maj,
    base_col_offset,
    column_index,
    col_repeat,
):
    col_block_stride = 0 if col_repeat == 1 else mem_tile_n
    if b_col_maj:
        return TensorAccessPattern(
            (N, K),
            offset=(base_col_offset + column_index * n) * K,
            sizes=[col_repeat, K_div_k, n, k],
            strides=[col_block_stride * K, k, K, 1],
        )
    return TensorAccessPattern(
        (K, N),
        offset=base_col_offset + column_index * n,
        sizes=[col_repeat, K_div_k, k, n],
        strides=[col_block_stride, k * N, N, 1],
    )


def _make_c_tile_tap(
    M,
    N,
    n,
    mem_tile_n,
    c_col_maj,
    row_offset,
    base_col_offset,
    row_repeat,
    col_repeat,
    row_block_stride,
    column_index,
    row_height,
):
    if c_col_maj:
        return TensorAccessPattern(
            (N, M),
            offset=(base_col_offset + column_index * n) * M + row_offset,
            sizes=[col_repeat, row_repeat, n, row_height],
            strides=[mem_tile_n * M, row_block_stride, M, 1],
        )
    return TensorAccessPattern(
        (M, N),
        offset=row_offset * N + base_col_offset + column_index * n,
        sizes=[row_repeat, col_repeat, row_height, n],
        strides=[row_block_stride * N, mem_tile_n, N, 1],
    )


def _uniform_worker_tile_counts(count: int, n_aie_cols: int):
    return tuple(tuple(count for _ in range(n_aie_cols)) for _ in range(4))


def plan_dynamic_gemm_fill_drain_tasks(
    *,
    M: int,
    K: int,
    N: int,
    m: int,
    k: int,
    n: int,
    n_aie_cols: int,
    b_col_maj: bool,
    c_col_maj: bool,
    separate_c_tiles: bool,
) -> DynamicGemmTransferPlan:
    if not separate_c_tiles:
        raise ValueError("Dynamic GEMM requires separate_c_tiles=True")
    if n_aie_cols not in (4, 8):
        raise ValueError("Dynamic GEMM supports n_aie_cols of 4 or 8")
    if M % m != 0:
        raise ValueError(f"M ({M}) must be divisible by m ({m})")
    if K % k != 0:
        raise ValueError(f"K ({K}) must be divisible by k ({k})")
    if N % n != 0:
        raise ValueError(f"N ({N}) must be divisible by n ({n})")

    n_aie_rows = 4
    mem_tile_m_C = m * n_aie_rows
    mem_tile_n = n * n_aie_cols
    K_div_k = K // k
    whole_row_blocks = M // mem_tile_m_C
    whole_col_blocks = N // mem_tile_n
    row_tail_rows = (M - whole_row_blocks * mem_tile_m_C) // m
    col_tail_cols = (N - whole_col_blocks * mem_tile_n) // n
    row_tail_offset = whole_row_blocks * mem_tile_m_C
    col_tail_offset = whole_col_blocks * mem_tile_n
    drain_row_height = m if c_col_maj else mem_tile_m_C

    phases: list[DynamicGemmPhasePlan] = []

    if whole_row_blocks > 0 and whole_col_blocks > 0:
        whole_tasks: list[DynamicGemmRuntimeTask] = []
        for row_block in range(whole_row_blocks):
            row_base = row_block * mem_tile_m_C
            for col_block in range(whole_col_blocks):
                col_base = col_block * mem_tile_n
                group_id = (row_block, col_block)
                for col in range(n_aie_cols):
                    compute_tiles = _column_compute_tiles(col, n_aie_rows)
                    whole_tasks.append(
                        DynamicGemmRuntimeTask(
                            tensor_name="C",
                            action="drain",
                            phase="whole",
                            group_id=group_id,
                            endpoint_index=col,
                            compute_tiles=compute_tiles,
                            tap=_make_c_tile_tap(
                                M,
                                N,
                                n,
                                mem_tile_n,
                                c_col_maj,
                                row_base,
                                col_base,
                                1,
                                1,
                                mem_tile_m_C,
                                col,
                                drain_row_height,
                            ),
                        )
                    )
                    if col < n_aie_rows:
                        whole_tasks.append(
                            DynamicGemmRuntimeTask(
                                tensor_name="A",
                                action="fill",
                                phase="whole",
                                group_id=group_id,
                                endpoint_index=col,
                                compute_tiles=_row_compute_tiles(col, n_aie_cols),
                                tap=_make_a_column_repeat_tap(
                                    M,
                                    K,
                                    row_base + col * m,
                                    m,
                                    k,
                                    K_div_k,
                                    1,
                                ),
                            )
                        )
                    whole_tasks.append(
                        DynamicGemmRuntimeTask(
                            tensor_name="B",
                            action="fill",
                            phase="whole",
                            group_id=group_id,
                            endpoint_index=col,
                            compute_tiles=compute_tiles,
                            tap=_make_b_column_block_tap(
                                K,
                                N,
                                k,
                                n,
                                K_div_k,
                                mem_tile_n,
                                b_col_maj,
                                col_base,
                                col,
                                1,
                            ),
                        )
                    )

        phases.append(
            DynamicGemmPhasePlan(
                name="whole",
                worker_tile_counts=_uniform_worker_tile_counts(
                    whole_row_blocks * whole_col_blocks, n_aie_cols
                ),
                tasks=tuple(whole_tasks),
            )
        )

    if row_tail_rows > 0 and whole_col_blocks > 0:
        row_tail_tasks: list[DynamicGemmRuntimeTask] = []
        sink_row_offset = row_tail_offset - (n_aie_rows - row_tail_rows) * m
        for col_block in range(whole_col_blocks):
            col_base = col_block * mem_tile_n
            for row in range(n_aie_rows):
                source_row = sink_row_offset + row * m
                row_tail_tasks.append(
                    DynamicGemmRuntimeTask(
                        tensor_name="A",
                        action="fill",
                        phase="row_tail",
                        group_id=col_block,
                        endpoint_index=row,
                        compute_tiles=_row_compute_tiles(row, n_aie_cols),
                        tap=_make_a_column_repeat_tap(
                            M, K, source_row, m, k, K_div_k, 1
                        ),
                    )
                )
            for col in range(n_aie_cols):
                compute_tiles = _column_compute_tiles(col, n_aie_rows)
                row_tail_tasks.append(
                    DynamicGemmRuntimeTask(
                        tensor_name="B",
                        action="fill",
                        phase="row_tail",
                        group_id=col_block,
                        endpoint_index=col,
                        compute_tiles=compute_tiles,
                        tap=_make_b_column_block_tap(
                            K,
                            N,
                            k,
                            n,
                            K_div_k,
                            mem_tile_n,
                            b_col_maj,
                            col_base,
                            col,
                            1,
                        ),
                    )
                )
                row_tail_tasks.append(
                    DynamicGemmRuntimeTask(
                        tensor_name="C",
                        action="drain",
                        phase="row_tail",
                        group_id=col_block,
                        endpoint_index=col,
                        compute_tiles=compute_tiles,
                        tap=_make_c_tile_tap(
                            M,
                            N,
                            n,
                            mem_tile_n,
                            c_col_maj,
                            sink_row_offset,
                            col_base,
                            1,
                            1,
                            mem_tile_m_C,
                            col,
                            drain_row_height,
                        ),
                    )
                )
        phases.append(
            DynamicGemmPhasePlan(
                name="row_tail",
                worker_tile_counts=_uniform_worker_tile_counts(
                    whole_col_blocks, n_aie_cols
                ),
                tasks=tuple(row_tail_tasks),
            )
        )

    if whole_row_blocks > 0 and col_tail_cols > 0:
        col_tail_tasks: list[DynamicGemmRuntimeTask] = []
        last_valid_col = col_tail_cols - 1
        for row_block in range(whole_row_blocks):
            row_base = row_block * mem_tile_m_C
            group_id = row_block
            for row in range(n_aie_rows):
                col_tail_tasks.append(
                    DynamicGemmRuntimeTask(
                        tensor_name="A",
                        action="fill",
                        phase="column_tail",
                        group_id=group_id,
                        endpoint_index=row,
                        compute_tiles=_row_compute_tiles(row, n_aie_cols),
                        tap=_make_a_column_repeat_tap(
                            M,
                            K,
                            row_base + row * m,
                            m,
                            k,
                            K_div_k,
                            1,
                        ),
                    )
                )
            for col in range(n_aie_cols):
                source_col = min(col, last_valid_col)
                compute_tiles = _column_compute_tiles(col, n_aie_rows)
                col_tail_tasks.append(
                    DynamicGemmRuntimeTask(
                        tensor_name="B",
                        action="fill",
                        phase="column_tail",
                        group_id=group_id,
                        endpoint_index=col,
                        compute_tiles=compute_tiles,
                        tap=_make_b_column_block_tap(
                            K,
                            N,
                            k,
                            n,
                            K_div_k,
                            mem_tile_n,
                            b_col_maj,
                            col_tail_offset,
                            source_col,
                            1,
                        ),
                    )
                )
            drain_column_order = tuple(range(col_tail_cols, n_aie_cols)) + tuple(
                range(col_tail_cols)
            )
            for col in drain_column_order:
                sink_col = min(col, last_valid_col)
                col_tail_tasks.append(
                    DynamicGemmRuntimeTask(
                        tensor_name="C",
                        action="drain",
                        phase="column_tail",
                        group_id=group_id,
                        endpoint_index=col,
                        compute_tiles=_column_compute_tiles(col, n_aie_rows),
                        tap=_make_c_tile_tap(
                            M,
                            N,
                            n,
                            mem_tile_n,
                            c_col_maj,
                            row_base,
                            col_tail_offset,
                            1,
                            1,
                            mem_tile_m_C,
                            sink_col,
                            drain_row_height,
                        ),
                    )
                )
        phases.append(
            DynamicGemmPhasePlan(
                name="column_tail",
                worker_tile_counts=_uniform_worker_tile_counts(
                    whole_row_blocks, n_aie_cols
                ),
                tasks=tuple(col_tail_tasks),
            )
        )

    if row_tail_rows > 0 and col_tail_cols > 0:
        corner_tasks: list[DynamicGemmRuntimeTask] = []
        last_valid_col = col_tail_cols - 1
        sink_row_offset = row_tail_offset - (n_aie_rows - row_tail_rows) * m
        for row in range(n_aie_rows):
            source_row = sink_row_offset + row * m
            corner_tasks.append(
                DynamicGemmRuntimeTask(
                    tensor_name="A",
                    action="fill",
                    phase="corner_tail",
                    group_id=0,
                    endpoint_index=row,
                    compute_tiles=_row_compute_tiles(row, n_aie_cols),
                    tap=_make_a_column_repeat_tap(M, K, source_row, m, k, K_div_k, 1),
                )
            )
        for col in range(n_aie_cols):
            source_col = min(col, last_valid_col)
            compute_tiles = _column_compute_tiles(col, n_aie_rows)
            corner_tasks.append(
                DynamicGemmRuntimeTask(
                    tensor_name="B",
                    action="fill",
                    phase="corner_tail",
                    group_id=0,
                    endpoint_index=col,
                    compute_tiles=compute_tiles,
                    tap=_make_b_column_block_tap(
                        K,
                        N,
                        k,
                        n,
                        K_div_k,
                        mem_tile_n,
                        b_col_maj,
                        col_tail_offset,
                        source_col,
                        1,
                    ),
                )
            )
        drain_column_order = tuple(range(col_tail_cols, n_aie_cols)) + tuple(
            range(col_tail_cols)
        )
        for col in drain_column_order:
            sink_col = min(col, last_valid_col)
            corner_tasks.append(
                DynamicGemmRuntimeTask(
                    tensor_name="C",
                    action="drain",
                    phase="corner_tail",
                    group_id=0,
                    endpoint_index=col,
                    compute_tiles=_column_compute_tiles(col, n_aie_rows),
                    tap=_make_c_tile_tap(
                        M,
                        N,
                        n,
                        mem_tile_n,
                        c_col_maj,
                        sink_row_offset,
                        col_tail_offset,
                        1,
                        1,
                        mem_tile_m_C,
                        sink_col,
                        drain_row_height,
                    ),
                )
            )
        phases.append(
            DynamicGemmPhasePlan(
                name="corner_tail",
                worker_tile_counts=_uniform_worker_tile_counts(1, n_aie_cols),
                tasks=tuple(corner_tasks),
            )
        )

    return DynamicGemmTransferPlan(phases=tuple(phases))


def my_matmul(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    c_col_maj,
    use_scalar,
    emulate_bf16_mmul_with_bfp16,
    prio_accuracy,
    separate_c_tiles,
    trace_size,
    archive=None,
    generate_taps=False,
):
    n_aie_rows = 4
    if n_aie_cols not in (4, 8):
        raise AssertionError("Dynamic GEMM supports n_aie_cols of 4 or 8")
    if M % m != 0:
        raise AssertionError(f"M ({M}) must be divisible by m ({m})")
    if N % n != 0:
        raise AssertionError(f"N ({N}) must be divisible by n ({n})")

    transfer_plan = plan_dynamic_gemm_fill_drain_tasks(
        M=M,
        K=K,
        N=N,
        m=m,
        k=k,
        n=n,
        n_aie_cols=n_aie_cols,
        b_col_maj=bool(b_col_maj),
        c_col_maj=bool(c_col_maj),
        separate_c_tiles=bool(separate_c_tiles),
    )
    if generate_taps:
        return transfer_plan.A_taps, transfer_plan.B_taps, transfer_plan.C_taps

    full_M = m * n_aie_rows
    whole_row_blocks = M // full_M
    row_tail_rows = (M - whole_row_blocks * full_M) // m
    if row_tail_rows > 0 and whole_row_blocks == 0:
        raise AssertionError(
            "Dynamic GEMM executable row/corner tail handling requires at least one full row block"
        )

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    if prio_accuracy:
        assert dtype_out_str == "bf16"
        use_larger_internal_buffer = True
        dtype_out_internal = str_to_dtype("f32")
    else:
        use_larger_internal_buffer = False

    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    if dev == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU has 4 columns")
    if dev == "npu2" and n_aie_cols > 8:
        raise AssertionError("Invalid configuration: NPU2 has 8 columns")
    if not use_scalar:
        assert m % r == 0
        assert k % s == 0
        assert n % t == 0
    assert K % k == 0

    mem_tile_m_A = m
    mem_tile_m_C = m * n_aie_rows
    fifo_depth = 2

    dev_ty = NPU1() if dev == "npu" else NPU2()

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(mem_tile_m_A * k,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(mem_tile_m_C * n,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    scalar_suffix = "_scalar" if use_scalar else ""
    archive_name = f"dynamic_gemm_{m}x{k}x{n}_archive.a" if archive is None else archive
    if use_larger_internal_buffer:
        fifo_depth_out = 1
        C_l1_ty_internal = np.ndarray[(m, n), np.dtype[dtype_out_internal]]
        convert_copy_kernel = Kernel(
            "convert_copy_f32_to_bf16",
            archive_name,
            [C_l1_ty_internal, C_l1_ty, np.int32],
        )
        zero_kernel = Kernel(
            f"zero{scalar_suffix}_f32", archive_name, [C_l1_ty_internal]
        )
        matmul_kernel = Kernel(
            f"matmul{scalar_suffix}_{dtype_in_str}_f32",
            archive_name,
            [A_l1_ty, B_l1_ty, C_l1_ty_internal],
        )
    else:
        fifo_depth_out = fifo_depth
        convert_copy_kernel = None
        zero_kernel = Kernel(
            f"zero{scalar_suffix}_{dtype_out_str}", archive_name, [C_l1_ty]
        )
        matmul_kernel = Kernel(
            f"matmul{scalar_suffix}_{dtype_in_str}_{dtype_out_str}",
            archive_name,
            [A_l1_ty, B_l1_ty, C_l1_ty],
        )

    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    A_l3l2_fifos = [
        ObjectFifo(A_l2_ty, name=f"A_L3L2_{row}", depth=fifo_depth)
        for row in range(n_aie_rows)
    ]
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    rtps = [
        [
            Buffer(
                np.ndarray[(2,), np.dtype[np.int32]],
                name=f"rtp{row}_{col}",
                initial_value=np.array([0, 0], dtype=np.int32),
                use_write_rtp=True,
            )
            for col in range(n_aie_cols)
        ]
        for row in range(n_aie_rows)
    ]
    worker_barriers = [
        [WorkerRuntimeBarrier(initial_value=0) for _ in range(n_aie_cols)]
        for _ in range(n_aie_rows)
    ]

    for row in range(n_aie_rows):
        dims_to_stream = [
            (m // r, r * k),
            (k // s, s),
            (r, k),
            (s, 1),
        ]
        A_l2l1_fifos[row] = (
            A_l3l2_fifos[row]
            .cons()
            .forward(
                obj_type=A_l1_ty,
                name=f"A_L2L1_{row}",
                dims_to_stream=dims_to_stream,
                placement=Tile(2 * row if n_aie_cols == 8 else row, 1),
            )
        )

    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        if b_col_maj:
            dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=dims_to_stream,
                placement=Tile(col, 1),
            )
        )

        if c_col_maj:
            dims_to_stream = [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
        else:
            dims_to_stream = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=dims_to_stream,
        )
        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                [m * n * row for row in range(n_aie_rows)],
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth_out] * n_aie_rows,
                placement=Tile(col, 1),
            )
        )
        for row in range(n_aie_rows):
            C_l1l2_fifos[row][col] = c_tmp_fifos[row]

    def core_fn(
        in_a,
        in_b,
        out_c,
        zero,
        matmul,
        convert_copy,
        my_rtp,
        barrier,
        elem_out_internal,
    ):
        for _ in range_(sys.maxsize):
            barrier.wait_for_value(1)
            rtp_K_div_k = my_rtp[0]
            rtp_n_tiles_per_core = my_rtp[1]
            loop = range(1)
            if rtp_n_tiles_per_core > 1:
                loop = range_(rtp_n_tiles_per_core)
            for _ in loop:
                if not use_larger_internal_buffer:
                    elem_out_internal = out_c.acquire(1)
                zero(elem_out_internal)

                for _ in range_(rtp_K_div_k):
                    elem_in_a = in_a.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out_internal)
                    in_a.release(1)
                    in_b.release(1)

                if use_larger_internal_buffer:
                    elem_out_transfer = out_c.acquire(1)
                    convert_copy(elem_out_internal, elem_out_transfer, m * n)
                    out_c.release(1)
                else:
                    out_c.release(1)
            barrier.wait_for_value(0)

    workers = []
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            tile_col, tile_row = core_tiles[row][col]
            acc_buffer = None
            if use_larger_internal_buffer:
                acc_buffer = Buffer(
                    type=C_l1_ty_internal, name=f"acc_buffer_{row}_{col}"
                )

            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[row][col].prod(),
                        zero_kernel,
                        matmul_kernel,
                        convert_copy_kernel,
                        rtps[row][col],
                        worker_barriers[row][col],
                        acc_buffer,
                    ],
                    placement=Tile(tile_col, tile_row),
                    stack_size=0xD00,
                )
            )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)

        def set_rtps(worker_counts):
            def _setter(*rtp_rows):
                for row, rtp_row in enumerate(rtp_rows):
                    for col, rtp in enumerate(rtp_row):
                        rtp[0] = K // k
                        rtp[1] = worker_counts[row][col]

            return _setter

        def _phase_group_order(tasks):
            group_order = []
            for task in tasks:
                if task.group_id not in group_order:
                    group_order.append(task.group_id)
            return group_order

        def _emit_task(task, task_group):
            if task.tensor_name == "A":
                row = task.compute_tiles[0][0]
                rt.fill(
                    A_l3l2_fifos[row].prod(),
                    A,
                    tap=task.tap,
                    task_group=task_group,
                    placement=Tile(2 * row if n_aie_cols == 8 else row, 0),
                )
            elif task.tensor_name == "B":
                col = task.compute_tiles[0][1]
                rt.fill(
                    B_l3l2_fifos[col].prod(),
                    B,
                    tap=task.tap,
                    task_group=task_group,
                    placement=Tile(col, 0),
                )
            else:
                col = task.compute_tiles[0][1]
                rt.drain(
                    C_l2l3_fifos[col].cons(),
                    C,
                    tap=task.tap,
                    wait=True,
                    task_group=task_group,
                    placement=Tile(col, 0),
                )

        for phase in transfer_plan.phases:
            if not phase.has_work:
                continue
            rt.inline_ops(set_rtps(phase.worker_tile_counts), rtps)
            for row in range(n_aie_rows):
                for col in range(n_aie_cols):
                    rt.set_barrier(worker_barriers[row][col], 1)
            for group_id in _phase_group_order(phase.tasks):
                tg = rt.task_group()
                for task in phase.tasks:
                    if task.group_id != group_id:
                        continue
                    _emit_task(task, tg)
                rt.finish_task_group(tg)
            for row in range(n_aie_rows):
                for col in range(n_aie_cols):
                    rt.set_barrier(worker_barriers[row][col], 0)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    main()
