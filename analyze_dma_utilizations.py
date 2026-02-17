# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import argparse
from pathlib import Path


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze patterns in MLIR files")
    parser.add_argument("file_path", help="Path to the MLIR file to analyze")

    args = parser.parse_args()
    mlir_file = args.file_path

    # Read the MLIR file
    with open(mlir_file, "r") as f:
        content = f.read()

    # Define regex patterns for each substring type
    patterns = {
        "{%mem_tile_X_Y}": r"\{%mem_tile_\d+_\d+\}",
        "(%shim_noc_tile_X_Y": r"\(%shim_noc_tile_\d+_\d+",
        "(%mem_tile_X_Y": r"\(%mem_tile_\d+_\d+",
        "{%shim_noc_tile_X_Y}": r"\{%shim_noc_tile_\d+_\d+\}",
    }
    pattern_descriptions = {
        "{%mem_tile_X_Y}": "Mem tile channels in",
        "(%shim_noc_tile_X_Y": "Shim NOC tile channels out",
        "(%mem_tile_X_Y": "Mem tile channels out",
        "{%shim_noc_tile_X_Y}": "Shim NOC tile channels in",
    }

    # Count occurrences for each unique X, Y value for each pattern
    from collections import defaultdict

    results = {}
    for pattern_name, regex_pattern in patterns.items():
        matches = re.findall(regex_pattern, content)
        xy_counts = defaultdict(int)

        # Extract X, Y values from each match and count occurrences
        for match in matches:
            # Extract X and Y using regex
            xy_match = re.search(r"(\d+)_(\d+)", match)
            if xy_match:
                x, y = xy_match.groups()
                xy_counts[(x, y)] += 1

        results[pattern_name] = xy_counts

    # Write results to a text file
    output_file = Path.cwd() / "dma_utilizations.log"

    with open(output_file, "w") as f:
        f.write("Pattern Analysis Results\n")
        f.write("=" * 50 + "\n\n")

        total_matches = 0
        for pattern_name, xy_counts in results.items():
            f.write(
                f"Pattern: {pattern_name}, Description: {pattern_descriptions.get(pattern_name, 'N/A')}\n"
            )
            f.write(f"Unique X, Y values: {len(xy_counts)}\n")
            f.write(f"Total count: {sum(xy_counts.values())}\n")
            f.write("X, Y value counts:\n")

            # Sort by X, Y for consistent output
            for (x, y), count in sorted(
                xy_counts.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
            ):
                f.write(f"  ({x}, {y}): {count}\n")

            f.write("\n")
            total_matches += sum(xy_counts.values())

        f.write("=" * 50 + "\n")
        f.write(f"Total matches across all patterns: {total_matches}\n")

    print("Analysis complete!")
    print("\nPattern Analysis Results:")
    print("=" * 50)
    for pattern_name, xy_counts in results.items():
        print(
            f"Pattern: {pattern_name}, Description: {pattern_descriptions.get(pattern_name, 'N/A')}"
        )
        print(f"Unique X, Y values: {len(xy_counts)}")
        print(f"Total count: {sum(xy_counts.values())}")
        print("X, Y value counts:")

        # Sort by X, Y for consistent output
        for (x, y), count in sorted(
            xy_counts.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
        ):
            print(f"  ({x}, {y}): {count}")

        print()
    print("=" * 50)
    print(f"Total matches across all patterns: {total_matches}")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
