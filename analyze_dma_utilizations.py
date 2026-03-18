# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

TILE_REF_RE = re.compile(r"%(?:mem_tile|shim_noc_tile|tile)_\d+_\d+")
OBJECTFIFO_RE = re.compile(
    r"aie\.objectfifo\s+@(?P<name>\w+)\((?P<src>.*?)\s*,\s*\{(?P<dsts>.*?)\}\s*,\s*(?P<depth>.*?)\)\s*:\s*!aie\.objectfifo<(?P<obj_type>.*?)>",
    re.DOTALL,
)
OBJECTFIFO_LINK_RE = re.compile(
    r"aie\.objectfifo\.link\s+\[(?P<srcs>.*?)\]\s*->\s*\[(?P<dsts>.*?)\]\((?P<src_offsets>.*?)\s+(?P<dst_offsets>.*?)\)",
    re.DOTALL,
)
AIE_DMA_CONFIG_START_RE = re.compile(
    r"aiex\.dma_configure_task_for\s+@(?P<name>\w+)\s*\{"
)
AIE_DMA_BD_RE = re.compile(r"aie\.dma_bd\(")
DMA_REPEAT_RE = re.compile(r"repeat_count\s*=\s*(\d+)\s*:\s*i32")
DMA_LEN_RE = re.compile(r"len\s*=\s*(\d+)\s*:\s*i32")
DMA_POSITIONAL_LEN_RE = re.compile(
    r"aie\.dma_bd\([^)]*?,\s*\d+\s*,\s*(\d+)\s*,",
    re.DOTALL,
)
AIE_FLOW_RE = re.compile(
    r"aie\.flow\(\s*(?P<src>%\w+_\d+_\d+)\s*,\s*DMA\s*:\s*(?P<src_dma>\d+)\s*,\s*"
    r"(?P<dst>%\w+_\d+_\d+)\s*,\s*DMA\s*:\s*(?P<dst_dma>\d+)\s*\)"
)


@dataclass(frozen=True)
class ObjectFifo:
    name: str
    src_tile: str
    dst_tiles: tuple[str, ...]
    obj_type: str
    producer_depth: int
    consumer_depth: int


@dataclass(frozen=True)
class Surface:
    name: str
    kind: str
    fifos: tuple[ObjectFifo, ...]


@dataclass(frozen=True)
class RuntimeDmaTask:
    name: str
    bd_defs: int
    repeat_count: int
    lens: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeDmaTaskSummary:
    name: str
    task_defs: int
    bd_defs: int
    effective_dispatches: int
    repeat_counts: tuple[int, ...]
    lens: tuple[int, ...]
    tile: str | None
    direction: str | None


def tile_sort_key(tile_ref: str) -> tuple[int, int, int]:
    kind_order = {
        "shim_noc_tile": 0,
        "mem_tile": 1,
        "tile": 2,
    }
    match = re.search(r"%(shim_noc_tile|mem_tile|tile)_(\d+)_(\d+)", tile_ref)
    if not match:
        return (99, 99, 99)
    kind, x, y = match.groups()
    return (kind_order[kind], int(x), int(y))


def classify_tile(tile_ref: str) -> str:
    if tile_ref.startswith("%mem_tile_"):
        return "mem_tile"
    if tile_ref.startswith("%shim_noc_tile_"):
        return "shim"
    return "core"


def parse_fifo_depth(depth_text: str) -> tuple[int, int]:
    values = [int(v) for v in re.findall(r"(\d+)\s*:\s*i32", depth_text)]
    if not values:
        return (1, 1)
    if len(values) == 1:
        return (values[0], values[0])
    return (values[0], values[1])


def parse_objectfifos(content: str) -> dict[str, ObjectFifo]:
    objectfifos: dict[str, ObjectFifo] = {}
    for match in OBJECTFIFO_RE.finditer(content):
        src_tiles = TILE_REF_RE.findall(match.group("src"))
        if len(src_tiles) != 1:
            continue
        dst_tiles = tuple(TILE_REF_RE.findall(match.group("dsts")))
        producer_depth, consumer_depth = parse_fifo_depth(match.group("depth"))
        objectfifos[match.group("name")] = ObjectFifo(
            name=match.group("name"),
            src_tile=src_tiles[0],
            dst_tiles=dst_tiles,
            obj_type=match.group("obj_type"),
            producer_depth=producer_depth,
            consumer_depth=consumer_depth,
        )
    return objectfifos


def parse_objectfifo_links(
    content: str,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    links: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for match in OBJECTFIFO_LINK_RE.finditer(content):
        srcs = tuple(re.findall(r"@(\w+)", match.group("srcs")))
        dsts = tuple(re.findall(r"@(\w+)", match.group("dsts")))
        links.append((srcs, dsts))
    return links


def classify_surface_kind(
    src_fifos: tuple[ObjectFifo, ...],
    dst_fifos: tuple[ObjectFifo, ...],
) -> str:
    if not src_fifos and not dst_fifos:
        return "empty"
    if not dst_fifos:
        return "direct"
    if len(src_fifos) > 1 and len(dst_fifos) == 1:
        return "join"
    if len(src_fifos) == 1 and len(dst_fifos) > 1:
        return "split"
    if len(src_fifos) == 1 and len(dst_fifos) == 1:
        if len(dst_fifos[0].dst_tiles) > 1:
            return "forward"
        return "relay"
    return "compound"


def build_surfaces(
    objectfifos: dict[str, ObjectFifo],
    links: list[tuple[tuple[str, ...], tuple[str, ...]]],
) -> list[Surface]:
    surfaces: list[Surface] = []
    linked_names: set[str] = set()
    for src_names, dst_names in links:
        src_fifos = tuple(
            objectfifos[name] for name in src_names if name in objectfifos
        )
        dst_fifos = tuple(
            objectfifos[name] for name in dst_names if name in objectfifos
        )
        if not src_fifos and not dst_fifos:
            continue
        linked_names.update(fifo.name for fifo in src_fifos + dst_fifos)
        kind = classify_surface_kind(src_fifos, dst_fifos)
        name = f"{'/'.join(src_names)} -> {'/'.join(dst_names)}"
        surfaces.append(
            Surface(
                name=name,
                kind=kind,
                fifos=src_fifos + dst_fifos,
            )
        )

    for fifo in objectfifos.values():
        if fifo.name in linked_names:
            continue
        surfaces.append(
            Surface(
                name=fifo.name,
                kind="direct",
                fifos=(fifo,),
            )
        )

    return surfaces


def build_counts(
    surfaces: list[Surface],
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, dict[str, int]]],
]:
    channel_counts = {"input": defaultdict(int), "output": defaultdict(int)}
    stress_counts = {"input": defaultdict(int), "output": defaultdict(int)}
    channel_details = {"input": defaultdict(list), "output": defaultdict(list)}
    stress_details = {"input": defaultdict(list), "output": defaultdict(list)}
    kind_counts = {
        "input": defaultdict(lambda: defaultdict(int)),
        "output": defaultdict(lambda: defaultdict(int)),
    }

    for surface in surfaces:
        output_channels: dict[str, int] = defaultdict(int)
        output_stress: dict[str, int] = defaultdict(int)
        input_channels: dict[str, int] = defaultdict(int)
        input_stress: dict[str, int] = defaultdict(int)

        for fifo in surface.fifos:
            fanout = len(fifo.dst_tiles)
            output_channels[fifo.src_tile] += 1
            output_stress[fifo.src_tile] += fanout
            for dst_tile in fifo.dst_tiles:
                input_channels[dst_tile] += 1
                input_stress[dst_tile] += 1

        for tile, count in output_channels.items():
            channel_counts["output"][tile] += count
            stress_counts["output"][tile] += output_stress[tile]
            kind_counts["output"][tile][surface.kind] += count
            channel_details["output"][tile].append(
                f"{surface.kind} {surface.name}: +{count} producer channel(s)"
            )
            stress_details["output"][tile].append(
                f"{surface.kind} {surface.name}: +{output_stress[tile]} fanout stress"
            )

        for tile, count in input_channels.items():
            channel_counts["input"][tile] += count
            stress_counts["input"][tile] += input_stress[tile]
            kind_counts["input"][tile][surface.kind] += count
            channel_details["input"][tile].append(
                f"{surface.kind} {surface.name}: +{count} consumer channel(s)"
            )
            stress_details["input"][tile].append(
                f"{surface.kind} {surface.name}: +{input_stress[tile]} consumer demand"
            )

    return channel_counts, stress_counts, channel_details, stress_details, kind_counts


def build_bd_estimates(
    surfaces: list[Surface],
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, dict[str, int]]],
]:
    bd_counts = {
        "input": defaultdict(int),
        "output": defaultdict(int),
        "total": defaultdict(int),
    }
    bd_details = {
        "input": defaultdict(list),
        "output": defaultdict(list),
    }
    bd_contributors = {
        "input": defaultdict(lambda: defaultdict(int)),
        "output": defaultdict(lambda: defaultdict(int)),
    }

    for surface in surfaces:
        output_depths: dict[str, int] = defaultdict(int)
        input_depths: dict[str, int] = defaultdict(int)

        for fifo in surface.fifos:
            output_depths[fifo.src_tile] += fifo.producer_depth
            for dst_tile in fifo.dst_tiles:
                input_depths[dst_tile] += fifo.consumer_depth

        for tile, depth in output_depths.items():
            bd_counts["output"][tile] += depth
            bd_counts["total"][tile] += depth
            bd_contributors["output"][tile][f"{surface.kind} {surface.name}"] += depth
            bd_details["output"][tile].append(
                f"{surface.kind} {surface.name}: +{depth} producer BD estimate"
            )

        for tile, depth in input_depths.items():
            bd_counts["input"][tile] += depth
            bd_counts["total"][tile] += depth
            bd_contributors["input"][tile][f"{surface.kind} {surface.name}"] += depth
            bd_details["input"][tile].append(
                f"{surface.kind} {surface.name}: +{depth} consumer BD estimate"
            )

    return bd_counts, bd_details, bd_contributors


def parse_aie_flows(
    content: str,
) -> list[tuple[str, int, str, int]]:
    return [
        (
            match.group("src"),
            int(match.group("src_dma")),
            match.group("dst"),
            int(match.group("dst_dma")),
        )
        for match in AIE_FLOW_RE.finditer(content)
    ]


def build_flow_channel_usage(
    flows: list[tuple[str, int, str, int]],
) -> dict[str, dict[str, dict[int, list[str]]]]:
    usage = {
        "input": defaultdict(lambda: defaultdict(list)),
        "output": defaultdict(lambda: defaultdict(list)),
    }
    for src, src_dma, dst, dst_dma in flows:
        usage["output"][src][src_dma].append(dst)
        usage["input"][dst][dst_dma].append(src)
    return usage


def extract_braced_block(content: str, open_brace_idx: int) -> tuple[str, int]:
    if open_brace_idx >= len(content) or content[open_brace_idx] != "{":
        raise ValueError("extract_braced_block requires an opening brace index")
    depth = 0
    start = open_brace_idx + 1
    idx = open_brace_idx
    while idx < len(content):
        char = content[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start:idx], idx + 1
        idx += 1
    raise ValueError("Unbalanced braces while parsing lowered MLIR")


def parse_dma_lengths(body: str) -> tuple[int, ...]:
    lengths = [int(value) for value in DMA_LEN_RE.findall(body)]
    if lengths:
        return tuple(lengths)
    return tuple(int(value) for value in DMA_POSITIONAL_LEN_RE.findall(body))


def parse_runtime_dma_tasks(content: str) -> list[RuntimeDmaTask]:
    tasks: list[RuntimeDmaTask] = []
    pos = 0
    while True:
        match = AIE_DMA_CONFIG_START_RE.search(content, pos)
        if match is None:
            break
        body, next_pos = extract_braced_block(content, match.end() - 1)
        bd_defs = len(AIE_DMA_BD_RE.findall(body))
        attrs = ""
        scan_pos = next_pos
        while scan_pos < len(content) and content[scan_pos].isspace():
            scan_pos += 1
        if scan_pos < len(content) and content[scan_pos] == "{":
            attrs, scan_pos = extract_braced_block(content, scan_pos)
        pos = scan_pos
        if bd_defs == 0:
            continue
        repeat_match = DMA_REPEAT_RE.search(attrs)
        repeat_count = int(repeat_match.group(1)) if repeat_match else 0
        lens = parse_dma_lengths(body)
        tasks.append(
            RuntimeDmaTask(
                name=match.group("name"),
                bd_defs=bd_defs,
                repeat_count=repeat_count,
                lens=lens,
            )
        )
    return tasks


def summarize_runtime_dma_tasks(
    tasks: list[RuntimeDmaTask],
    objectfifos: dict[str, ObjectFifo],
) -> list[RuntimeDmaTaskSummary]:
    grouped: dict[str, list[RuntimeDmaTask]] = defaultdict(list)
    for task in tasks:
        grouped[task.name].append(task)

    summaries: list[RuntimeDmaTaskSummary] = []
    for name, task_group in grouped.items():
        fifo = objectfifos.get(name)
        tile = None
        direction = None
        if fifo is not None:
            if classify_tile(fifo.src_tile) == "shim":
                tile = fifo.src_tile
                direction = "output"
            else:
                shim_dsts = [
                    dst for dst in fifo.dst_tiles if classify_tile(dst) == "shim"
                ]
                if shim_dsts:
                    tile = shim_dsts[0]
                    direction = "input"

        repeat_counts = tuple(sorted({task.repeat_count for task in task_group}))
        lens = tuple(sorted({length for task in task_group for length in task.lens}))
        bd_defs = sum(task.bd_defs for task in task_group)
        effective_dispatches = sum(
            task.bd_defs * (task.repeat_count + 1) for task in task_group
        )
        summaries.append(
            RuntimeDmaTaskSummary(
                name=name,
                task_defs=len(task_group),
                bd_defs=bd_defs,
                effective_dispatches=effective_dispatches,
                repeat_counts=repeat_counts,
                lens=lens,
                tile=tile,
                direction=direction,
            )
        )

    summaries.sort(key=lambda summary: (-summary.bd_defs, summary.name))
    return summaries


def build_lowered_runtime_bd_counts(
    summaries: list[RuntimeDmaTaskSummary],
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, dict[str, list[str]]],
]:
    counts = {
        "input": defaultdict(int),
        "output": defaultdict(int),
        "input_effective": defaultdict(int),
        "output_effective": defaultdict(int),
    }
    details = {
        "input": defaultdict(list),
        "output": defaultdict(list),
    }

    for summary in summaries:
        if summary.tile is None or summary.direction is None:
            continue
        counts[summary.direction][summary.tile] += summary.bd_defs
        counts[f"{summary.direction}_effective"][
            summary.tile
        ] += summary.effective_dispatches
        repeat_text = ",".join(str(value) for value in summary.repeat_counts)
        lens_text = (
            ",".join(str(value) for value in summary.lens) if summary.lens else "?"
        )
        details[summary.direction][summary.tile].append(
            f"{summary.name}: task_defs={summary.task_defs}, bd_defs={summary.bd_defs}, "
            f"effective_dispatches={summary.effective_dispatches}, repeats=[{repeat_text}], lens=[{lens_text}]"
        )

    return counts, details


def tile_limit(tile_ref: str, direction: str, args: argparse.Namespace) -> int | None:
    tile_kind = classify_tile(tile_ref)
    if tile_kind == "mem_tile":
        return (
            args.memtile_s2mm_limit if direction == "input" else args.memtile_mm2s_limit
        )
    if tile_kind == "shim":
        return args.shim_s2mm_limit if direction == "input" else args.shim_mm2s_limit
    return None


def format_count_section(
    title: str,
    counts: dict[str, dict[str, int]],
    details: dict[str, dict[str, list[str]]] | None = None,
    args: argparse.Namespace | None = None,
) -> str:
    lines = [title, "=" * len(title), ""]

    tiles = sorted(
        set(counts["input"].keys()) | set(counts["output"].keys()),
        key=tile_sort_key,
    )
    for tile in tiles:
        in_count = counts["input"].get(tile, 0)
        out_count = counts["output"].get(tile, 0)
        limit_note = ""
        if args is not None:
            in_limit = tile_limit(tile, "input", args)
            out_limit = tile_limit(tile, "output", args)
            notes = []
            if in_limit is not None and in_count > in_limit:
                notes.append(f"input>{in_limit}")
            if out_limit is not None and out_count > out_limit:
                notes.append(f"output>{out_limit}")
            if notes:
                limit_note = f"  OVER_LIMIT[{', '.join(notes)}]"
        lines.append(f"{tile}: input={in_count}, output={out_count}{limit_note}")
        if details is not None:
            for direction in ("input", "output"):
                contributors = details[direction].get(tile, [])
                if not contributors:
                    continue
                lines.append(f"  {direction} contributors:")
                for contributor in contributors:
                    lines.append(f"    - {contributor}")
        lines.append("")

    return "\n".join(lines)


def format_link_section(
    links: list[tuple[tuple[str, ...], tuple[str, ...]]],
    objectfifos: dict[str, ObjectFifo],
) -> str:
    lines = ["Objectfifo Links", "================", ""]
    for srcs, dsts in links:
        src_desc = []
        for src in srcs:
            fifo = objectfifos.get(src)
            if fifo is None:
                src_desc.append(src)
            else:
                src_desc.append(f"{src}({fifo.src_tile} -> {len(fifo.dst_tiles)} dst)")
        dst_desc = []
        for dst in dsts:
            fifo = objectfifos.get(dst)
            if fifo is None:
                dst_desc.append(dst)
            else:
                dst_desc.append(f"{dst}({fifo.src_tile} -> {len(fifo.dst_tiles)} dst)")
        lines.append(f"[{', '.join(src_desc)}] -> [{', '.join(dst_desc)}]")
    lines.append("")
    return "\n".join(lines)


def format_surface_section(surfaces: list[Surface]) -> str:
    lines = ["Logical Transport Surfaces", "==========================", ""]
    for surface in surfaces:
        fifo_names = ", ".join(fifo.name for fifo in surface.fifos)
        src_tiles = ", ".join(
            sorted({fifo.src_tile for fifo in surface.fifos}, key=tile_sort_key)
        )
        dst_tiles = ", ".join(
            sorted(
                {tile for fifo in surface.fifos for tile in fifo.dst_tiles},
                key=tile_sort_key,
            )
        )
        lines.append(f"{surface.kind}: {surface.name}")
        lines.append(f"  fifos: {fifo_names}")
        lines.append(f"  src tiles: {src_tiles}")
        lines.append(f"  dst tiles: {dst_tiles}")
    lines.append("")
    return "\n".join(lines)


def format_kind_mix_section(
    kind_counts: dict[str, dict[str, dict[str, int]]],
    counts: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> str:
    lines = ["Per-Tile Surface Class Mix", "==========================", ""]
    tiles = sorted(
        set(counts["input"].keys()) | set(counts["output"].keys()),
        key=tile_sort_key,
    )
    for tile in tiles:
        lines.append(tile)
        tile_kind = classify_tile(tile)
        for direction in ("input", "output"):
            kinds = kind_counts[direction].get(tile, {})
            if not kinds:
                continue
            mix = ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
            limit = tile_limit(tile, direction, args)
            flags = []
            if tile_kind == "mem_tile":
                relay_count = kinds.get("relay", 0)
                split_count = kinds.get("split", 0)
                join_count = kinds.get("join", 0)
                forward_count = kinds.get("forward", 0)
                non_direct_classes = sum(
                    1 for kind, count in kinds.items() if kind != "direct" and count > 0
                )
                if relay_count >= 4:
                    flags.append(f"relay_heavy={relay_count}")
                if split_count + join_count >= 3:
                    flags.append(f"boundary_heavy={split_count + join_count}")
                if non_direct_classes >= 3:
                    flags.append(f"mixed_classes={non_direct_classes}")
                if (
                    limit is not None
                    and counts[direction].get(tile, 0) >= max(limit - 1, 0)
                    and relay_count > 0
                    and (forward_count > 0 or split_count > 0 or join_count > 0)
                ):
                    flags.append("mixed_near_limit")
            flag_note = f"  FLAGS[{', '.join(flags)}]" if flags else ""
            lines.append(f"  {direction}: {mix}{flag_note}")
        lines.append("")
    return "\n".join(lines)


def format_bd_section(
    counts: dict[str, dict[str, int]],
    details: dict[str, dict[str, list[str]]],
    bd_limit: int,
) -> str:
    lines = ["Estimated BD Pressure", "=====================", ""]
    tiles = sorted(set(counts["total"].keys()), key=tile_sort_key)
    for tile in tiles:
        in_bd = counts["input"].get(tile, 0)
        out_bd = counts["output"].get(tile, 0)
        total_bd = counts["total"].get(tile, 0)
        note = ""
        if total_bd > bd_limit:
            note = f"  OVER_LIMIT[total>{bd_limit}]"
        elif total_bd >= max(bd_limit - 4, 0):
            note = f"  NEAR_LIMIT[total={total_bd}/{bd_limit}]"
        lines.append(
            f"{tile}: input_bd={in_bd}, output_bd={out_bd}, total_bd={total_bd}{note}"
        )
        for direction in ("input", "output"):
            contributors = details[direction].get(tile, [])
            if not contributors:
                continue
            lines.append(f"  {direction} contributors:")
            for contributor in contributors:
                lines.append(f"    - {contributor}")
        lines.append("")
    return "\n".join(lines)


def format_runtime_dma_task_section(
    summaries: list[RuntimeDmaTaskSummary],
) -> str:
    lines = ["Lowered Runtime DMA Tasks", "========================", ""]
    for summary in summaries:
        repeat_text = ",".join(str(value) for value in summary.repeat_counts)
        lens_text = (
            ",".join(str(value) for value in summary.lens) if summary.lens else "?"
        )
        placement = ""
        if summary.tile is not None and summary.direction is not None:
            placement = f", tile={summary.tile}, direction={summary.direction}"
        lines.append(
            f"@{summary.name}: task_defs={summary.task_defs}, bd_defs={summary.bd_defs}, "
            f"effective_dispatches={summary.effective_dispatches}, repeats=[{repeat_text}], "
            f"lens=[{lens_text}]{placement}"
        )
    lines.append("")
    return "\n".join(lines)


def format_lowered_runtime_bd_section(
    counts: dict[str, dict[str, int]],
    details: dict[str, dict[str, list[str]]],
) -> str:
    lines = ["Lowered Runtime Shim BD Load", "============================", ""]
    tiles = sorted(
        set(counts["input"].keys()) | set(counts["output"].keys()),
        key=tile_sort_key,
    )
    for tile in tiles:
        lines.append(
            f"{tile}: input_bd_defs={counts['input'].get(tile, 0)}, "
            f"output_bd_defs={counts['output'].get(tile, 0)}, "
            f"input_effective_dispatches={counts['input_effective'].get(tile, 0)}, "
            f"output_effective_dispatches={counts['output_effective'].get(tile, 0)}"
        )
        for direction in ("input", "output"):
            contributors = details[direction].get(tile, [])
            if not contributors:
                continue
            lines.append(f"  {direction} contributors:")
            for contributor in contributors:
                lines.append(f"    - {contributor}")
    lines.append("")
    return "\n".join(lines)


def format_flow_channel_section(
    usage: dict[str, dict[str, dict[int, list[str]]]],
) -> str:
    lines = ["Pre-BD DMA Flow Channels", "========================", ""]
    tiles = sorted(
        set(usage["input"].keys()) | set(usage["output"].keys()),
        key=tile_sort_key,
    )
    for tile in tiles:
        lines.append(tile)
        for direction in ("input", "output"):
            channels = usage[direction].get(tile, {})
            if not channels:
                continue
            for dma_channel, peers in sorted(channels.items()):
                peer_text = ", ".join(peers)
                lines.append(
                    f"  {direction} DMA {dma_channel}: {len(peers)} flow(s) -> [{peer_text}]"
                )
        lines.append("")
    return "\n".join(lines)


def format_bd_delta_section(
    primary_counts: dict[str, dict[str, int]],
    primary_contributors: dict[str, dict[str, dict[str, int]]],
    compare_counts: dict[str, dict[str, int]],
    compare_contributors: dict[str, dict[str, dict[str, int]]],
    bd_limit: int,
    compare_label: str,
) -> str:
    lines = [
        f"Estimated BD Delta vs {compare_label}",
        "=" * (23 + len(compare_label)),
        "",
    ]
    tiles = sorted(
        set(primary_counts["total"].keys()) | set(compare_counts["total"].keys()),
        key=lambda tile: (
            -abs(
                primary_counts["total"].get(tile, 0)
                - compare_counts["total"].get(tile, 0)
            ),
            tile_sort_key(tile),
        ),
    )
    for tile in tiles:
        input_delta = primary_counts["input"].get(tile, 0) - compare_counts[
            "input"
        ].get(tile, 0)
        output_delta = primary_counts["output"].get(tile, 0) - compare_counts[
            "output"
        ].get(tile, 0)
        total_delta = primary_counts["total"].get(tile, 0) - compare_counts[
            "total"
        ].get(tile, 0)
        if input_delta == 0 and output_delta == 0 and total_delta == 0:
            continue
        near_limit = ""
        current_total = primary_counts["total"].get(tile, 0)
        if current_total > bd_limit:
            near_limit = f"  OVER_LIMIT[current>{bd_limit}]"
        elif current_total >= max(bd_limit - 4, 0):
            near_limit = f"  NEAR_LIMIT[current={current_total}/{bd_limit}]"
        lines.append(
            f"{tile}: input_bd_delta={input_delta:+}, output_bd_delta={output_delta:+}, total_bd_delta={total_delta:+}{near_limit}"
        )
        for direction in ("input", "output"):
            primary = primary_contributors[direction].get(tile, {})
            compare = compare_contributors[direction].get(tile, {})
            labels = set(primary.keys()) | set(compare.keys())
            deltas = []
            for label in labels:
                delta = primary.get(label, 0) - compare.get(label, 0)
                if delta != 0:
                    deltas.append((abs(delta), delta, label))
            if not deltas:
                continue
            deltas.sort(reverse=True)
            lines.append(f"  {direction} delta contributors:")
            for _, delta, label in deltas[:6]:
                lines.append(f"    - {label}: {delta:+}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MLIR objectfifo topology and estimate DMA channel demand. "
            "The main estimate treats each objectfifo producer as one output "
            "channel and each destination as one input channel. A secondary "
            "fanout-stress view highlights high-fanout producers."
        )
    )
    parser.add_argument("file_path", help="Path to the MLIR file to analyze")
    parser.add_argument(
        "--compare-to",
        help="Optional second MLIR file to diff against the primary report",
    )
    parser.add_argument(
        "--output",
        default="dma_utilizations.log",
        help="Path to the report file to write",
    )
    parser.add_argument("--memtile-s2mm-limit", type=int, default=6)
    parser.add_argument("--memtile-mm2s-limit", type=int, default=6)
    parser.add_argument("--shim-s2mm-limit", type=int, default=2)
    parser.add_argument("--shim-mm2s-limit", type=int, default=2)
    parser.add_argument("--bd-limit", type=int, default=48)
    parser.add_argument(
        "--show-links",
        action="store_true",
        help="Include parsed objectfifo.link topology in the report",
    )
    args = parser.parse_args()

    mlir_path = Path(args.file_path)
    content = mlir_path.read_text()
    objectfifos = parse_objectfifos(content)
    links = parse_objectfifo_links(content)
    surfaces = build_surfaces(objectfifos, links)
    (
        channel_counts,
        stress_counts,
        channel_details,
        stress_details,
        kind_counts,
    ) = build_counts(surfaces)
    bd_counts, bd_details, bd_contributors = build_bd_estimates(surfaces)
    flows = parse_aie_flows(content)
    flow_usage = build_flow_channel_usage(flows)
    runtime_dma_tasks = parse_runtime_dma_tasks(content)
    runtime_dma_summaries = summarize_runtime_dma_tasks(runtime_dma_tasks, objectfifos)
    lowered_runtime_counts, lowered_runtime_details = build_lowered_runtime_bd_counts(
        runtime_dma_summaries
    )

    report_sections = [
        f"DMA Utilization Analysis: {mlir_path}",
        "",
        format_surface_section(surfaces),
        format_count_section(
            "Estimated DMA Channel Counts",
            channel_counts,
            channel_details,
            args,
        ),
        format_kind_mix_section(kind_counts, channel_counts, args),
        format_bd_section(bd_counts, bd_details, args.bd_limit),
        format_flow_channel_section(flow_usage),
        format_count_section(
            "Fanout Stress Estimate",
            stress_counts,
            stress_details,
            args,
        ),
    ]
    if runtime_dma_summaries:
        report_sections.extend(
            [
                format_runtime_dma_task_section(runtime_dma_summaries),
                format_lowered_runtime_bd_section(
                    lowered_runtime_counts, lowered_runtime_details
                ),
            ]
        )
    if args.show_links:
        report_sections.append(format_link_section(links, objectfifos))
    if args.compare_to:
        compare_path = Path(args.compare_to)
        compare_content = compare_path.read_text()
        compare_objectfifos = parse_objectfifos(compare_content)
        compare_links = parse_objectfifo_links(compare_content)
        compare_surfaces = build_surfaces(compare_objectfifos, compare_links)
        compare_bd_counts, _, compare_bd_contributors = build_bd_estimates(
            compare_surfaces
        )
        report_sections.append(
            format_bd_delta_section(
                bd_counts,
                bd_contributors,
                compare_bd_counts,
                compare_bd_contributors,
                args.bd_limit,
                str(compare_path),
            )
        )

    report = "\n".join(report_sections).rstrip() + "\n"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.write_text(report)

    print(report)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
