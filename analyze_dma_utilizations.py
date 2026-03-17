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


@dataclass(frozen=True)
class ObjectFifo:
    name: str
    src_tile: str
    dst_tiles: tuple[str, ...]
    obj_type: str


@dataclass(frozen=True)
class Surface:
    name: str
    kind: str
    fifos: tuple[ObjectFifo, ...]


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


def parse_objectfifos(content: str) -> dict[str, ObjectFifo]:
    objectfifos: dict[str, ObjectFifo] = {}
    for match in OBJECTFIFO_RE.finditer(content):
        src_tiles = TILE_REF_RE.findall(match.group("src"))
        if len(src_tiles) != 1:
            continue
        dst_tiles = tuple(TILE_REF_RE.findall(match.group("dsts")))
        objectfifos[match.group("name")] = ObjectFifo(
            name=match.group("name"),
            src_tile=src_tiles[0],
            dst_tiles=dst_tiles,
            obj_type=match.group("obj_type"),
        )
    return objectfifos


def parse_objectfifo_links(content: str) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
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
        src_fifos = tuple(objectfifos[name] for name in src_names if name in objectfifos)
        dst_fifos = tuple(objectfifos[name] for name in dst_names if name in objectfifos)
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


def tile_limit(tile_ref: str, direction: str, args: argparse.Namespace) -> int | None:
    tile_kind = classify_tile(tile_ref)
    if tile_kind == "mem_tile":
        return args.memtile_s2mm_limit if direction == "input" else args.memtile_mm2s_limit
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
        src_tiles = ", ".join(sorted({fifo.src_tile for fifo in surface.fifos}, key=tile_sort_key))
        dst_tiles = ", ".join(
            sorted({tile for fifo in surface.fifos for tile in fifo.dst_tiles}, key=tile_sort_key)
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
            mix = ", ".join(
                f"{kind}={count}" for kind, count in sorted(kinds.items())
            )
            limit = tile_limit(tile, direction, args)
            flags = []
            if tile_kind == "mem_tile":
                relay_count = kinds.get("relay", 0)
                split_count = kinds.get("split", 0)
                join_count = kinds.get("join", 0)
                forward_count = kinds.get("forward", 0)
                non_direct_classes = sum(
                    1
                    for kind, count in kinds.items()
                    if kind != "direct" and count > 0
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
        "--output",
        default="dma_utilizations.log",
        help="Path to the report file to write",
    )
    parser.add_argument("--memtile-s2mm-limit", type=int, default=6)
    parser.add_argument("--memtile-mm2s-limit", type=int, default=6)
    parser.add_argument("--shim-s2mm-limit", type=int, default=2)
    parser.add_argument("--shim-mm2s-limit", type=int, default=2)
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
        format_count_section(
            "Fanout Stress Estimate",
            stress_counts,
            stress_details,
            args,
        ),
    ]
    if args.show_links:
        report_sections.append(format_link_section(links, objectfifos))

    report = "\n".join(report_sections).rstrip() + "\n"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.write_text(report)

    print(report)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
