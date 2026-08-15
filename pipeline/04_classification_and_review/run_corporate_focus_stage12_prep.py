#!/usr/bin/env python3
# ruff: noqa: E402
"""Prepare the corporate-focused subset and Colab package for Stage 1/2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from cross_source_microtopic_common import configure_logging
from microtopic_posthoc_merge_common import (
    CORPORATE_FOCUS_PAIR_OUTPUT_ROOT,
    CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT,
    CORPORATE_FOCUS_STAGE12_INPUT_ROOT,
    MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED,
    MERGE_FIRST_GROUP_REVIEW_MULTIASPECT_OUTPUT_ROOT,
    RAW_MICRO_ROOT_MULTIASPECT,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
TOPIC_MODELING_DIR = PIPELINE_ROOT / "02_topic_modeling"
DESCRIPTION_DIR = PIPELINE_ROOT / "03_topic_description_and_interpretation"
REVIEW_DIR = PIPELINE_ROOT / "04_classification_and_review"
CROSS_SOURCE_DIR = PIPELINE_ROOT / "05_cross_source_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-micro-root", type=Path, default=RAW_MICRO_ROOT_MULTIASPECT)
    parser.add_argument(
        "--group-review-root",
        type=Path,
        default=MERGE_FIRST_GROUP_REVIEW_MULTIASPECT_OUTPUT_ROOT,
    )
    parser.add_argument("--merged-micro-root", type=Path, default=MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED)
    parser.add_argument("--pair-root", type=Path, default=CORPORATE_FOCUS_PAIR_OUTPUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT)
    parser.add_argument("--stage12-output-root", type=Path, default=CORPORATE_FOCUS_STAGE12_INPUT_ROOT)
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--embedding-model-name", type=str, default=None)
    parser.add_argument("--max-chunks-per-year", type=int, default=5)
    parser.add_argument("--skip-colab-package", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run_script(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"[corporate-focus] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def workbook_path(group_review_root: Path) -> Path:
    candidates = sorted(group_review_root.glob("*.xlsx"))
    if not candidates:
        raise SystemExit(f"No workbook found in {group_review_root}")
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if "multiaspect" in candidate.name:
            return candidate
    return candidates[0]


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    reviewed_workbook = workbook_path(args.group_review_root)

    run_script(
        TOPIC_MODELING_DIR / "materialize_microtopic_group_review_mapping.py",
        [
            "--output-root",
            str(args.group_review_root),
            "--workbook",
            str(reviewed_workbook),
            "--log-level",
            args.log_level,
        ],
    )

    mapping_csv = args.group_review_root / "microtopic_to_merged_group.csv"
    run_script(
        TOPIC_MODELING_DIR / "build_merged_microtopic_root.py",
        [
            "--micro-root",
            str(args.raw_micro_root),
            "--mapping-csv",
            str(mapping_csv),
            "--output-root",
            str(args.merged_micro_root),
            "--log-level",
            args.log_level,
        ],
    )

    run_script(
        CROSS_SOURCE_DIR / "build_cross_source_microtopic_profiles.py",
        [
            "--micro-root",
            str(args.merged_micro_root),
            "--output-root",
            str(args.pair_root),
            "--log-level",
            args.log_level,
        ],
    )

    embedding_args = [
        "--output-root",
        str(args.pair_root),
        "--log-level",
        args.log_level,
    ]
    if args.embedding_model_name:
        embedding_args.extend(["--model-name", args.embedding_model_name])
    run_script(
        CROSS_SOURCE_DIR / "embed_cross_source_microtopic_profiles.py",
        embedding_args,
    )

    run_script(
        CROSS_SOURCE_DIR / "match_cross_source_microtopics.py",
        [
            "--output-root",
            str(args.pair_root),
            "--log-level",
            args.log_level,
        ],
    )

    run_script(
        REVIEW_DIR / "build_corporate_focus_review.py",
        [
            "--merged-micro-root",
            str(args.merged_micro_root),
            "--pair-root",
            str(args.pair_root),
            "--output-root",
            str(args.review_root),
            "--similarity-threshold",
            str(args.similarity_threshold),
            "--log-level",
            args.log_level,
        ],
    )

    run_script(
        DESCRIPTION_DIR / "build_corporate_focus_stage12_inputs.py",
        [
            "--micro-root",
            str(args.merged_micro_root),
            "--review-root",
            str(args.review_root),
            "--output-root",
            str(args.stage12_output_root),
            "--max-chunks-per-year",
            str(args.max_chunks_per_year),
            "--log-level",
            args.log_level,
        ],
    )

    if not args.skip_colab_package:
        run_script(
            DESCRIPTION_DIR / "build_corporate_focus_colab_package.py",
            [
                "--input-root",
                str(args.stage12_output_root),
            ],
        )

    print(args.stage12_output_root)


if __name__ == "__main__":
    main()
