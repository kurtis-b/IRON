# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import product

COMPUTE_COLS = tuple(range(8))
COMPUTE_ROWS = tuple(range(2, 6))


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def cardinal_neighbors(tile: tuple[int, int]) -> list[tuple[int, int]]:
    c, r = tile
    candidates = [(c, r + 1), (c + 1, r), (c, r - 1), (c - 1, r)]
    return [
        (cc, rr) for (cc, rr) in candidates if cc in COMPUTE_COLS and rr in COMPUTE_ROWS
    ]


def reduction_neighbors_nes(tile: tuple[int, int]) -> list[tuple[int, int]]:
    """Directed neighbors for down-reduction links (north/east/south only)."""
    c, r = tile
    candidates = [(c, r + 1), (c + 1, r), (c, r - 1)]
    return [
        (cc, rr) for (cc, rr) in candidates if cc in COMPUTE_COLS and rr in COMPUTE_ROWS
    ]


def find_ffn_layout(
    parallel_heads: int,
    max_non_neighbor_down_to_ln2: int = 2,
    mha_output_tile: tuple[int, int] | None = None,
    require_mha_ln_neighbor: bool = False,
    restrict_down_reduction_to_nes: bool = True,
    restrict_down_to_ln2_to_nes: bool = False,
) -> dict:
    all_compute_tiles = {(c, r) for c in COMPUTE_COLS for r in COMPUTE_ROWS}
    mha_tiles = {(c, r) for c in range(parallel_heads) for r in COMPUTE_ROWS}
    free_tiles = all_compute_tiles - mha_tiles

    # Down tile must be north/east/south of paired up tile.
    # Equivalently, up tile is south/west/north of down tile.
    def allowed_up_tiles_for_down(down_tile: tuple[int, int]) -> list[tuple[int, int]]:
        c, r = down_tile
        candidates = [(c, r - 1), (c - 1, r), (c, r + 1)]
        return [
            (cc, rr)
            for (cc, rr) in candidates
            if cc in COMPUTE_COLS and rr in COMPUTE_ROWS
        ]

    def enumerate_neighbor_chains(
        available_tiles: set[tuple[int, int]],
        length: int,
        next_hops_fn,
    ) -> list[list[tuple[int, int]]]:
        chains: list[list[tuple[int, int]]] = []
        for start in sorted(available_tiles):
            stack = [(start, [start], {start})]
            while stack:
                cur, path, visited = stack.pop()
                if len(path) == length:
                    chains.append(path)
                    continue
                for nxt in sorted(next_hops_fn(cur)):
                    if nxt not in available_tiles or nxt in visited:
                        continue
                    stack.append((nxt, path + [nxt], visited | {nxt}))
        return chains

    def pair_orientation_penalty(
        up_tile: tuple[int, int], down_tile: tuple[int, int]
    ) -> int:
        dc = down_tile[0] - up_tile[0]
        dr = down_tile[1] - up_tile[1]
        # Target geometry preference: up on the left, down on the right.
        if (dc, dr) == (1, 0):
            return 0
        # Accept N/S placement as fallback under routing pressure.
        if (dc, dr) in ((0, 1), (0, -1)):
            return 1
        return 5

    ln1_candidates = sorted(free_tiles)
    if not ln1_candidates:
        raise ValueError("No free tiles available for LN1 placement after MHA mapping.")

    best = None
    for ln1_tile in ln1_candidates:
        if (
            require_mha_ln_neighbor
            and mha_output_tile is not None
            and manhattan_distance(ln1_tile, mha_output_tile) != 1
        ):
            continue
        available = set(free_tiles) - {ln1_tile}
        max_chain_len = min(6, len(available))
        if max_chain_len <= 0:
            continue
        down_chain_hops_fn = (
            reduction_neighbors_nes
            if restrict_down_reduction_to_nes
            else cardinal_neighbors
        )
        for chain_len in range(max_chain_len, 0, -1):
            down_chains = enumerate_neighbor_chains(
                available, chain_len, down_chain_hops_fn
            )
            for down_tiles in down_chains:
                down_set = set(down_tiles)
                up_options = []
                feasible = True
                for down_tile in down_tiles:
                    options = [
                        t
                        for t in allowed_up_tiles_for_down(down_tile)
                        if t in free_tiles and t != ln1_tile and t not in down_set
                    ]
                    if not options:
                        feasible = False
                        break
                    up_options.append(options)
                if not feasible:
                    continue

                for up_choice in product(*up_options):
                    if len(set(up_choice)) != chain_len:
                        continue
                    up_tiles = list(up_choice)
                    used_with_down = {ln1_tile, *up_tiles, *down_tiles}
                    ln2_candidates = [t for t in free_tiles if t not in used_with_down]
                    root_down = down_tiles[-1]
                    reduction_edges = [
                        (down_tiles[i], down_tiles[i + 1]) for i in range(chain_len - 1)
                    ]
                    down_to_ln2 = [root_down]

                    for ln2_tile in ln2_candidates:
                        if (
                            restrict_down_to_ln2_to_nes
                            and ln2_tile not in reduction_neighbors_nes(root_down)
                        ):
                            continue
                        non_neighbor_down = [
                            d
                            for d in down_to_ln2
                            if manhattan_distance(d, ln2_tile) != 1
                        ]
                        if len(non_neighbor_down) > max_non_neighbor_down_to_ln2:
                            continue

                        mha_ln_dist = (
                            manhattan_distance(ln1_tile, mha_output_tile)
                            if mha_output_tile is not None
                            else 0
                        )
                        ln_row_score = -(ln1_tile[1] + ln2_tile[1])
                        ffn_row_score = sum(r for (_, r) in up_tiles) + sum(
                            r for (_, r) in down_tiles
                        )
                        orientation_score = sum(
                            pair_orientation_penalty(up_tile, down_tile)
                            for up_tile, down_tile in zip(up_tiles, down_tiles)
                        )
                        down_cols = [c for (c, _) in down_tiles]
                        up_cols = [c for (c, _) in up_tiles]
                        down_col_span = max(down_cols) - min(down_cols)
                        up_col_span = max(up_cols) - min(up_cols)
                        col_gap_score = -sum(
                            down_tile[0] - up_tile[0]
                            for up_tile, down_tile in zip(up_tiles, down_tiles)
                        )
                        score = (
                            -chain_len,
                            orientation_score,
                            down_col_span,
                            up_col_span,
                            ln_row_score,
                            ffn_row_score,
                            col_gap_score,
                            len(non_neighbor_down),
                            manhattan_distance(root_down, ln2_tile),
                            mha_ln_dist,
                            -sum(c for (c, _) in down_tiles),
                            ln2_tile[0],
                            ln2_tile[1],
                            ln1_tile[0],
                            ln1_tile[1],
                        )
                        if best is None or score < best["score"]:
                            best = {
                                "score": score,
                                "mha_tiles": sorted(mha_tiles),
                                "free_tiles": sorted(free_tiles),
                                "ln1_tile": ln1_tile,
                                "up_tiles": up_tiles,
                                "down_tiles": down_tiles,
                                "ln2_tile": ln2_tile,
                                "down_root_tile": root_down,
                                "down_reduction_edges": reduction_edges,
                                "down_to_ln2_tiles": down_to_ln2,
                                "non_neighbor_down_to_ln2": non_neighbor_down,
                            }

    if best is None:
        if require_mha_ln_neighbor and mha_output_tile is not None:
            raise ValueError(
                "Unable to find FFN layout satisfying LN1/UP/DOWN/LN2 constraints "
                f"with required MHA->LN1 neighbor relation (mha_output={mha_output_tile})."
            )
        raise ValueError(
            "Unable to find FFN layout satisfying neighbor constraints for LN1/UP/DOWN/LN2."
        )

    return best
