# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import product
from pathlib import Path
import re

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


def find_ffn_layout(
    parallel_heads: int,
    max_non_neighbor_down_to_ln2: int = 2,
    mha_output_tile: tuple[int, int] | None = None,
    require_mha_ln_neighbor: bool = False,
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
        available_tiles: set[tuple[int, int]], length: int
    ) -> list[list[tuple[int, int]]]:
        chains: list[list[tuple[int, int]]] = []
        for start in sorted(available_tiles):
            stack = [(start, [start], {start})]
            while stack:
                cur, path, visited = stack.pop()
                if len(path) == length:
                    chains.append(path)
                    continue
                for nxt in sorted(cardinal_neighbors(cur)):
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
        max_chain_len = min(3, len(available))
        if max_chain_len <= 0:
            continue
        for chain_len in range(max_chain_len, 0, -1):
            down_chains = enumerate_neighbor_chains(available, chain_len)
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


def validate_layout_constraints(
    parallel_heads: int,
    layout: dict,
    max_non_neighbor_down_to_ln2: int = 2,
    mha_output_tile: tuple[int, int] | None = None,
    require_mha_ln_neighbor: bool = False,
) -> list[str]:
    errors: list[str] = []

    ln1_tile = tuple(layout["ln1_tile"])
    up_tiles = [tuple(t) for t in layout["up_tiles"]]
    down_tiles = [tuple(t) for t in layout["down_tiles"]]
    ln2_tile = tuple(layout["ln2_tile"])
    down_to_ln2 = [tuple(t) for t in layout.get("down_to_ln2_tiles", [])]
    down_reduction_edges = [
        (tuple(src), tuple(dst))
        for (src, dst) in layout.get("down_reduction_edges", [])
    ]

    if mha_output_tile is None:
        mha_output_tile = (parallel_heads - 1, 5)

    if require_mha_ln_neighbor and manhattan_distance(mha_output_tile, ln1_tile) != 1:
        errors.append(
            "MHA->LN1 producer/consumer are not neighboring: "
            f"mha_output={mha_output_tile}, ln1={ln1_tile}"
        )

    if len(up_tiles) != len(down_tiles):
        errors.append(
            "up/down stream counts mismatch: "
            f"up={len(up_tiles)}, down={len(down_tiles)}"
        )
    else:
        for idx, (up_tile, down_tile) in enumerate(zip(up_tiles, down_tiles)):
            if manhattan_distance(up_tile, down_tile) != 1:
                errors.append(
                    f"UP[{idx}]->DOWN[{idx}] is non-neighbor: "
                    f"up={up_tile}, down={down_tile}"
                )
                continue
            # Design intent: down tile is north/east/south of its paired up tile.
            dc = down_tile[0] - up_tile[0]
            dr = down_tile[1] - up_tile[1]
            if (dc, dr) not in [(0, 1), (1, 0), (0, -1)]:
                errors.append(
                    f"UP[{idx}]->DOWN[{idx}] is not in N/E/S relation: "
                    f"up={up_tile}, down={down_tile}, delta={(dc, dr)}"
                )

    down_set = set(down_tiles)
    down_to_ln2_set = set(down_to_ln2)

    for d in down_to_ln2:
        if d not in down_set:
            errors.append(f"down_to_ln2 includes tile not in down_tiles: {d}")

    src_to_dst: dict[tuple[int, int], tuple[int, int]] = {}
    for src, dst in down_reduction_edges:
        if src not in down_set or dst not in down_set:
            errors.append(f"Invalid reduction edge outside down set: {src}->{dst}")
            continue
        if manhattan_distance(src, dst) != 1:
            errors.append(f"Reduction edge is non-neighbor: {src}->{dst}")
        if src in src_to_dst:
            errors.append(
                f"Down tile has multiple reduction destinations: "
                f"{src}->{src_to_dst[src]} and {src}->{dst}"
            )
        src_to_dst[src] = dst

    # Every down tile must either stream to LN2 directly or reduce into one that does.
    for d in down_tiles:
        cur = d
        visited = set()
        covered = False
        while True:
            if cur in down_to_ln2_set:
                covered = True
                break
            if cur in visited:
                break
            visited.add(cur)
            if cur not in src_to_dst:
                break
            cur = src_to_dst[cur]
        if not covered:
            errors.append(
                "Down tile is not reduced/connected to LN2 stream set: "
                f"{d}, down_to_ln2={sorted(down_to_ln2_set)}"
            )

    non_neighbor_down_to_ln2 = [
        d for d in down_to_ln2 if manhattan_distance(d, ln2_tile) != 1
    ]
    if len(non_neighbor_down_to_ln2) > max_non_neighbor_down_to_ln2:
        errors.append(
            "Too many non-neighbor down->LN2 streams: "
            f"{len(non_neighbor_down_to_ln2)} > {max_non_neighbor_down_to_ln2} "
            f"(non-neighbor={non_neighbor_down_to_ln2}, ln2={ln2_tile})"
        )

    return errors


_OBJECTFIFO_SIMPLE_RE = re.compile(
    r"^\s*aie\.objectfifo @(?P<name>[A-Za-z0-9_]+)\("
    r"(?P<src>%[A-Za-z0-9_]+), \{(?P<dsts>[^}]*)\},"
)
_TOKEN_RE = re.compile(r"%[A-Za-z0-9_]+")
_COMPUTE_TILE_RE = re.compile(r"^%tile_(\d+)_(\d+)$")


def _parse_compute_tile(token: str) -> tuple[int, int] | None:
    token = token.strip()
    match = _COMPUTE_TILE_RE.match(token)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)))


def parse_simple_objectfifo_endpoints(
    mlir_path: str | Path,
) -> dict[str, dict]:
    mlir_text = Path(mlir_path).read_text(encoding="utf-8")
    fifos: dict[str, dict] = {}

    for line in mlir_text.splitlines():
        match = _OBJECTFIFO_SIMPLE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        src_token = match.group("src").strip()
        dst_tokens = [tok.strip() for tok in _TOKEN_RE.findall(match.group("dsts"))]
        fifos[name] = {
            "name": name,
            "source_token": src_token,
            "source_tile": _parse_compute_tile(src_token),
            "destination_tokens": dst_tokens,
            "destination_tiles": [_parse_compute_tile(tok) for tok in dst_tokens],
        }
    return fifos


def validate_mlir_mapping_constraints(
    mlir_path: str | Path,
    max_non_neighbor_down_to_ln2: int = 2,
    require_mha_ln_neighbor: bool = False,
) -> dict:
    errors: list[str] = []
    fifos = parse_simple_objectfifo_endpoints(mlir_path)

    out_o = fifos.get("outO")
    out_ln_fifos = [
        fifo
        for name, fifo in sorted(fifos.items())
        if name == "outLN" or name.startswith("outLNFfn")
    ]
    mem_out_ln_fifos = [
        fifo
        for name, fifo in sorted(fifos.items())
        if name == "memOutLN" or name.startswith("memOutLNFfn")
    ]
    out_ln = fifos.get("outLN")
    out_ln2 = fifos.get("outLN2")
    ffn_up_fifos = [
        fifo for name, fifo in sorted(fifos.items()) if name.startswith("ffnUpOut")
    ]

    if out_o is None:
        errors.append("Missing objectfifo `outO` in MLIR.")
        mha_output_tile = None
        ln1_tile = None
    else:
        mha_output_tile = out_o["source_tile"]
        out_o_dsts = [t for t in out_o["destination_tiles"] if t is not None]
        ln1_tile = out_o_dsts[0] if out_o_dsts else None
        if mha_output_tile is None:
            errors.append("`outO` source is not a compute tile.")
        if ln1_tile is None:
            errors.append("`outO` does not target a compute tile.")
        if (
            require_mha_ln_neighbor
            and mha_output_tile is not None
            and ln1_tile is not None
        ):
            if manhattan_distance(mha_output_tile, ln1_tile) != 1:
                errors.append(
                    "MHA->LN1 is non-neighbor in MLIR: "
                    f"mha_output={mha_output_tile}, ln1={ln1_tile}"
                )

    if out_ln is None:
        if not out_ln_fifos:
            errors.append("Missing objectfifo `outLN*` in MLIR.")
    elif out_ln["source_tile"] is None:
        errors.append("`outLN` source is not a compute tile.")
    for fifo in out_ln_fifos:
        src = fifo["source_tile"]
        if src is None:
            errors.append(f"`{fifo['name']}` source is not a compute tile.")

    # Prefer memOutLN* (memtile->up) for realized up-core consumers.
    if mem_out_ln_fifos:
        up_tiles = []
        for fifo in mem_out_ln_fifos:
            up_tiles.extend([t for t in fifo["destination_tiles"] if t is not None])
    elif out_ln_fifos:
        up_tiles = []
        for fifo in out_ln_fifos:
            up_tiles.extend([t for t in fifo["destination_tiles"] if t is not None])
    else:
        up_tiles = []
    up_tiles = sorted(set(up_tiles))

    if ln1_tile is not None:
        if not up_tiles:
            errors.append("No LN1->FFN-up compute destinations found.")
        for up_tile in up_tiles:
            if manhattan_distance(ln1_tile, up_tile) != 1:
                errors.append(
                    "LN1->UP is non-neighbor in MLIR: " f"ln1={ln1_tile}, up={up_tile}"
                )

    # Derive UP->DOWN edges from ffnUpOut* objectfifos.
    up_to_down_edges: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    down_tiles_set: set[tuple[int, int]] = set()
    for fifo in ffn_up_fifos:
        src = fifo["source_tile"]
        if src is None:
            errors.append(f"`{fifo['name']}` source is not a compute tile.")
            continue
        dest_tiles = [t for t in fifo["destination_tiles"] if t is not None]
        if not dest_tiles:
            errors.append(f"`{fifo['name']}` has no compute-tile destinations.")
            continue
        if up_tiles and src not in up_tiles:
            errors.append(
                f"`{fifo['name']}` source {src} is not in LN1 up tiles {up_tiles}."
            )
        for dst in dest_tiles:
            up_to_down_edges.append((src, dst, fifo["name"]))
            down_tiles_set.add(dst)

    if not up_to_down_edges:
        errors.append("No UP->DOWN edges detected from `ffnUpOut*` objectfifos.")

    for src, dst, fifo_name in up_to_down_edges:
        if manhattan_distance(src, dst) != 1:
            errors.append(
                f"UP->DOWN is non-neighbor in MLIR via `{fifo_name}`: "
                f"up={src}, down={dst}"
            )

    if up_tiles:
        up_sources = {src for src, _, _ in up_to_down_edges}
        missing_up = sorted(set(up_tiles) - up_sources)
        if missing_up:
            errors.append(
                "Some UP tiles do not feed any DOWN tile in MLIR: " f"{missing_up}"
            )

    if out_ln2 is None:
        errors.append("Missing objectfifo `outLN2` in MLIR.")
        ln2_tile = None
    else:
        ln2_tile = out_ln2["source_tile"]
        if ln2_tile is None:
            errors.append("`outLN2` source is not a compute tile.")

    down_tiles = sorted(down_tiles_set)
    reduction_edges: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    direct_down_to_ln2: set[tuple[int, int]] = set()
    if ln2_tile is not None and down_tiles:
        for fifo in fifos.values():
            src = fifo["source_tile"]
            if src is None or src not in down_tiles_set:
                continue
            for dst in [t for t in fifo["destination_tiles"] if t is not None]:
                if dst == ln2_tile:
                    direct_down_to_ln2.add(src)
                elif dst in down_tiles_set:
                    reduction_edges.append((src, dst, fifo["name"]))

    # Every DOWN tile should reach LN2 either directly or through down-to-down reduction.
    if ln2_tile is not None and down_tiles:
        adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {
            d: set() for d in down_tiles_set
        }
        for src, dst, _ in reduction_edges:
            adjacency.setdefault(src, set()).add(dst)
        for src in direct_down_to_ln2:
            adjacency.setdefault(src, set()).add(ln2_tile)

        for down_tile in down_tiles:
            stack = [down_tile]
            visited: set[tuple[int, int]] = set()
            reaches_ln2 = False
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                for nxt in adjacency.get(cur, set()):
                    if nxt == ln2_tile:
                        reaches_ln2 = True
                        break
                    stack.append(nxt)
                if reaches_ln2:
                    break
            if not reaches_ln2:
                errors.append(
                    "DOWN tile has no path to LN2 in MLIR reduction graph: "
                    f"{down_tile}"
                )

    non_neighbor_direct = []
    if ln2_tile is not None:
        non_neighbor_direct = sorted(
            [d for d in direct_down_to_ln2 if manhattan_distance(d, ln2_tile) != 1]
        )
        if len(non_neighbor_direct) > max_non_neighbor_down_to_ln2:
            errors.append(
                "Too many non-neighbor direct DOWN->LN2 streams in MLIR: "
                f"{len(non_neighbor_direct)} > {max_non_neighbor_down_to_ln2} "
                f"(non-neighbor={non_neighbor_direct}, ln2={ln2_tile})"
            )

    return {
        "mlir_path": str(mlir_path),
        "mha_output_tile": mha_output_tile,
        "ln1_tile": ln1_tile,
        "up_tiles": up_tiles,
        "down_tiles": down_tiles,
        "ln2_tile": ln2_tile,
        "up_to_down_edges": up_to_down_edges,
        "reduction_edges": reduction_edges,
        "direct_down_to_ln2": sorted(direct_down_to_ln2),
        "non_neighbor_direct_down_to_ln2": non_neighbor_direct,
        "errors": errors,
    }
