#!/usr/bin/env python3
# ruff: noqa: E402
"""Create the manual review workbook consumed by microtopic merge materialization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from cross_source_microtopic_common import configure_logging
from microtopic_posthoc_merge_common import (
    MERGE_FIRST_GROUP_REVIEW_MULTIASPECT_OUTPUT_ROOT,
    RAW_MICRO_ROOT_MULTIASPECT,
    ensure_directory,
    sanitize_frame_for_excel,
    workbook_autofit,
    write_json,
)


GROUP_SUMMARY_COLUMNS = [
    "subgroup",
    "proposed_group_id",
    "group_decision",
    "group_review_notes",
]
GROUP_MEMBER_COLUMNS = [
    "subgroup",
    "source",
    "assigned_label",
    "proposed_group_id",
    "micro_topic_id",
    "manual_split_group_id",
    "manual_member_notes",
    "topic_name_original",
    "topic_size",
    "topic_representation",
    "representative_texts",
]
SINGLETON_COLUMNS = [
    "subgroup",
    "source",
    "assigned_label",
    "micro_topic_id",
    "topic_name_original",
    "topic_size",
    "topic_representation",
    "representative_texts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=RAW_MICRO_ROOT_MULTIASPECT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MERGE_FIRST_GROUP_REVIEW_MULTIASPECT_OUTPUT_ROOT,
    )
    parser.add_argument("--workbook-name", default="microtopic_merge_first_group_review_multiaspect.xlsx")
    parser.add_argument("--max-representative-texts", type=int, default=3)
    parser.add_argument("--max-representative-chars", type=int, default=500)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def compact_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def representative_text_map(
    path: Path,
    max_texts: int,
    max_chars: int,
) -> dict[int, str]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or not {"micro_topic_id", "text"}.issubset(frame.columns):
        return {}
    frame["micro_topic_id"] = pd.to_numeric(frame["micro_topic_id"], errors="coerce")
    frame = frame.dropna(subset=["micro_topic_id"]).copy()
    frame["micro_topic_id"] = frame["micro_topic_id"].astype(int)
    result: dict[int, str] = {}
    for topic_id, group in frame.groupby("micro_topic_id", dropna=False):
        texts = [
            compact_text(value, max_chars)
            for value in group["text"].fillna("").astype(str).head(max_texts)
            if str(value).strip()
        ]
        result[int(topic_id)] = " || ".join(texts)
    return result


def build_singletons(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subgroup_dirs = sorted(
        path for path in args.micro_root.iterdir() if path.is_dir() and path.name != "summary"
    )
    for subgroup_dir in subgroup_dirs:
        manifest_path = subgroup_dir / "manifest.json"
        topic_info_path = subgroup_dir / "topic_info.csv"
        if not manifest_path.exists() or not topic_info_path.exists():
            continue
        manifest = pd.read_json(manifest_path, typ="series")
        topic_info = pd.read_csv(topic_info_path)
        required = {"Topic", "Name", "Count"}
        missing = sorted(required - set(topic_info.columns))
        if missing:
            raise SystemExit(f"{topic_info_path} is missing required columns: {missing}")

        representative = representative_text_map(
            subgroup_dir / "representative_docs.csv",
            max_texts=args.max_representative_texts,
            max_chars=args.max_representative_chars,
        )
        topic_info["Topic"] = pd.to_numeric(topic_info["Topic"], errors="coerce")
        topic_info = topic_info.dropna(subset=["Topic"]).copy()
        topic_info["Topic"] = topic_info["Topic"].astype(int)
        topic_info = topic_info.loc[topic_info["Topic"] != -1].copy()
        for row in topic_info.itertuples(index=False):
            topic_id = int(row.Topic)
            rows.append(
                {
                    "subgroup": subgroup_dir.name,
                    "source": str(manifest.get("source", "")),
                    "assigned_label": str(manifest.get("assigned_label", "")),
                    "micro_topic_id": topic_id,
                    "topic_name_original": str(row.Name),
                    "topic_size": int(row.Count),
                    "topic_representation": str(getattr(row, "Representation", "")),
                    "representative_texts": representative.get(topic_id, ""),
                }
            )
    return pd.DataFrame(rows, columns=SINGLETON_COLUMNS).sort_values(
        ["source", "assigned_label", "subgroup", "micro_topic_id"]
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)
    if not args.micro_root.exists():
        raise SystemExit(f"Microtopic root does not exist: {args.micro_root}")

    group_summary = pd.DataFrame(columns=GROUP_SUMMARY_COLUMNS)
    group_members = pd.DataFrame(columns=GROUP_MEMBER_COLUMNS)
    singletons = build_singletons(args)
    if singletons.empty:
        raise SystemExit(f"No non-outlier microtopics found under {args.micro_root}")

    instructions = pd.DataFrame(
        [
            {
                "step": 1,
                "instruction": "Review every row in singletons using the label, terms, size, and representative texts.",
            },
            {
                "step": 2,
                "instruction": "For topics that should be reviewed as one possible merge, remove their rows from singletons and copy them to group_members with the same proposed_group_id.",
            },
            {
                "step": 3,
                "instruction": "Add one group_summary row per proposed group and set group_decision to accept, reject, or split.",
            },
            {
                "step": 4,
                "instruction": "For split decisions, fill manual_split_group_id for every member. Leave it blank for accept or reject.",
            },
            {
                "step": 5,
                "instruction": "Save the workbook and run materialize_microtopic_group_review_mapping.py with --workbook pointing to it.",
            },
        ]
    )

    workbook_path = args.output_root / args.workbook_name
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, frame in {
            "group_summary": group_summary,
            "group_members": group_members,
            "singletons": singletons,
            "instructions": instructions,
        }.items():
            cleaned = sanitize_frame_for_excel(frame)
            cleaned.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook_autofit(writer, sheet_name, cleaned)

    group_summary.to_csv(args.output_root / "proposed_group_summary.csv", index=False)
    group_members.to_csv(args.output_root / "proposed_group_members.csv", index=False)
    singletons.to_csv(args.output_root / "singleton_topics.csv", index=False)
    write_json(
        args.output_root / "merge_review_workbook_manifest.json",
        {
            "micro_root": str(args.micro_root),
            "microtopic_count": int(len(singletons)),
            "initial_state": "all_non_outlier_microtopics_are_singletons_pending_manual_review",
            "outputs": {
                "workbook": str(workbook_path),
                "group_summary_csv": str(args.output_root / "proposed_group_summary.csv"),
                "group_members_csv": str(args.output_root / "proposed_group_members.csv"),
                "singletons_csv": str(args.output_root / "singleton_topics.csv"),
            },
        },
    )
    print(workbook_path)


if __name__ == "__main__":
    main()
