#!/usr/bin/env python3
"""Render Fig. 5-style document-topic maps for retained corporate-focus topics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from umap import UMAP

from workflow_common import load_embedding_memmap


PIPELINE_ROOT = Path("/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline")
DEFAULT_MODEL_ROOT = PIPELINE_ROOT / "outputs" / "bertopic_micro_merged_multiaspect_reviewed"
DEFAULT_SELECTED_TOPICS = (
    PIPELINE_ROOT
    / "outputs"
    / "corporate_focus_stage12_colab_drive_with_overrides"
    / "data"
    / "selected_micro_topics.csv"
)
DEFAULT_GROUP_METADATA = (
    PIPELINE_ROOT
    / "outputs"
    / "corporate_focus_review_with_overrides"
    / "corporate_focus_all_groups_optional.csv"
)
DEFAULT_PREVIOUS_REVIEW_DECISIONS = (
    PIPELINE_ROOT
    / "outputs"
    / "corporate_focus_review_with_overrides"
    / "corporate_focus_commented_decision_rows.csv"
)
DEFAULT_EXCLUSION_DECISIONS = (
    PIPELINE_ROOT
    / "outputs"
    / "corporate_external_group_exclusion_review"
    / "group_exclusion_decisions.csv"
)
DEFAULT_CATALOG = PIPELINE_ROOT / "catalog" / "six_topic_discourse_catalog.csv"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "outputs" / "paper_tables" / "corporate_focus_document_topic_maps"
DEFAULT_EMBEDDING_FILE = Path("/home/thiago/1_Supervised_BERTopic/2_dataset/filtered_embeddings.f32")
DEFAULT_EMBEDDING_META = Path("/home/thiago/1_Supervised_BERTopic/2_dataset/filtered_embeddings.meta.json")

MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
SOURCE_ORDER = ["academic", "media", "corporate"]
SOURCE_MARKERS = {"academic": "o", "media": "^", "corporate": "s"}
SOURCE_COLORS = {"academic": "#3569B8", "media": "#D8842A", "corporate": "#2C8C62"}
SOURCE_LABELS = {"academic": "Academic", "media": "Media", "corporate": "Corporate"}
REQUIRED_DOC_COLUMNS = [
    "chunk_id",
    "source_doc_id",
    "document_id",
    "doc_id",
    "source",
    "assigned_label",
    "year",
    "micro_topic_id",
    "raw_micro_topic_id",
    "final_merge_group_id",
    "merged_micro_topic_id",
    "embedding_row_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--selected-topics", type=Path, default=DEFAULT_SELECTED_TOPICS)
    parser.add_argument("--group-metadata", type=Path, default=DEFAULT_GROUP_METADATA)
    parser.add_argument("--previous-review-decisions", type=Path, default=DEFAULT_PREVIOUS_REVIEW_DECISIONS)
    parser.add_argument("--exclusion-decisions", type=Path, default=DEFAULT_EXCLUSION_DECISIONS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--embedding-meta", type=Path, default=DEFAULT_EMBEDDING_META)
    parser.add_argument("--macro-topics", nargs="*", default=MACRO_TOPIC_ORDER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--max-labels", type=int, default=12)
    return parser.parse_args()


def nice_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.facecolor": "#FAFAF8",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#D9D9D9",
            "grid.linestyle": ":",
            "axes.edgecolor": "#C8C8C8",
        }
    )


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip()


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_macro_topic_names(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    return dict(zip(frame["topic_code"].astype(str), frame["topic_name"].astype(str)))


def load_excluded_group_keys(path: Path) -> set[tuple[str, str]]:
    frame = read_csv_or_empty(path)
    if frame.empty:
        return set()
    decision_col = "_normalized_review_decision" if "_normalized_review_decision" in frame.columns else "review_decision"
    if decision_col not in frame.columns:
        return set()
    excluded = frame.loc[frame[decision_col].map(clean_text).str.lower().eq("exclude")].copy()
    if excluded.empty:
        return set()
    return set(zip(excluded["subgroup"].astype(str), excluded["final_merge_group_id"].astype(str)))


def load_previous_delete_group_keys(path: Path) -> set[tuple[str, str]]:
    frame = read_csv_or_empty(path)
    if frame.empty:
        return set()
    decision_col = "_normalized_review_decision" if "_normalized_review_decision" in frame.columns else "review_decision"
    required = {decision_col, "subgroup_noncorporate", "final_merge_group_id_noncorporate"}
    if not required.issubset(frame.columns):
        return set()
    deleted = frame.loc[frame[decision_col].map(clean_text).str.lower().eq("delete")].copy()
    if deleted.empty:
        return set()
    return set(
        zip(
            deleted["subgroup_noncorporate"].astype(str),
            deleted["final_merge_group_id_noncorporate"].astype(str),
        )
    )


def load_group_labels(path: Path) -> pd.DataFrame:
    frame = read_csv_or_empty(path)
    if frame.empty:
        return pd.DataFrame(columns=["subgroup", "final_merge_group_id", "group_label_refined", "group_summary"])
    required = {"subgroup", "final_merge_group_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Group metadata missing required columns: {missing}")

    label_col = "topic_label_refined" if "topic_label_refined" in frame.columns else "topic_name_original"
    summary_col = "overall_summary" if "overall_summary" in frame.columns else label_col
    labels = (
        frame.assign(
            subgroup=frame["subgroup"].astype(str),
            final_merge_group_id=frame["final_merge_group_id"].astype(str),
            _label=frame[label_col].map(clean_text),
            _summary=frame[summary_col].map(clean_text),
        )
        .sort_values(["subgroup", "final_merge_group_id"])
        .groupby(["subgroup", "final_merge_group_id"], dropna=False)
        .agg(
            group_label_refined=("_label", lambda values: next((value for value in values if value), "")),
            group_summary=("_summary", lambda values: next((value for value in values if value), "")),
        )
        .reset_index()
    )
    return labels


def load_retained_groups(args: argparse.Namespace, macro_topic_names: dict[str, str]) -> tuple[pd.DataFrame, int]:
    selected = pd.read_csv(args.selected_topics)
    metadata = pd.read_csv(args.group_metadata)

    selected_required = {"subgroup", "final_merge_group_id"}
    selected_missing = sorted(selected_required - set(selected.columns))
    if selected_missing:
        raise ValueError(f"Selected topics missing required columns: {selected_missing}")

    required = {"macro_topic", "source", "subgroup", "final_merge_group_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Group metadata missing required columns: {missing}")

    selected_group_keys = set(
        zip(
            selected["subgroup"].astype(str),
            selected["final_merge_group_id"].astype(str),
        )
    )
    metadata = metadata.copy()
    metadata["subgroup"] = metadata["subgroup"].astype(str)
    metadata["source"] = metadata["source"].astype(str)
    metadata["macro_topic"] = metadata["macro_topic"].astype(str)
    metadata["final_merge_group_id"] = metadata["final_merge_group_id"].astype(str)
    metadata["_group_key"] = list(zip(metadata["subgroup"], metadata["final_merge_group_id"]))

    excluded_keys = load_excluded_group_keys(args.exclusion_decisions)
    previous_delete_keys = load_previous_delete_group_keys(args.previous_review_decisions)
    removed_keys = excluded_keys | previous_delete_keys
    retained = metadata.loc[~metadata["_group_key"].isin(removed_keys)].copy()
    retained = retained[["macro_topic", "source", "subgroup", "final_merge_group_id"]].drop_duplicates()

    labels = load_group_labels(args.group_metadata)
    retained = retained.merge(labels, on=["subgroup", "final_merge_group_id"], how="left")
    retained["group_label_refined"] = retained["group_label_refined"].fillna("")
    retained["group_summary"] = retained["group_summary"].fillna("")
    retained["macro_topic_name"] = retained["macro_topic"].map(macro_topic_names).fillna("")
    retained["source_order"] = retained["source"].map({source: idx for idx, source in enumerate(SOURCE_ORDER)}).fillna(99)
    retained["macro_order"] = retained["macro_topic"].map({topic: idx for idx, topic in enumerate(MACRO_TOPIC_ORDER)}).fillna(99)
    retained = retained.sort_values(["macro_order", "source_order", "subgroup", "final_merge_group_id"]).reset_index(drop=True)
    return retained, len(selected_group_keys)


def read_subgroup_documents(subgroup_dir: Path) -> pd.DataFrame:
    path = subgroup_dir / "document_topics.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing document topics: {path}")
    schema_columns = set(pq.read_schema(path).names)
    columns = [column for column in REQUIRED_DOC_COLUMNS if column in schema_columns]
    return pd.read_parquet(path, columns=columns).copy()


def load_documents_for_groups(model_root: Path, groups: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for subgroup, subgroup_groups in groups.groupby("subgroup", sort=True):
        docs = read_subgroup_documents(model_root / subgroup)
        docs["subgroup"] = subgroup
        docs["final_merge_group_id"] = docs["final_merge_group_id"].astype(str)
        allowed = set(subgroup_groups["final_merge_group_id"].astype(str))
        docs = docs.loc[docs["final_merge_group_id"].isin(allowed)].copy()
        if "micro_topic_id" in docs.columns:
            docs["micro_topic_id"] = pd.to_numeric(docs["micro_topic_id"], errors="coerce")
            docs = docs.loc[docs["micro_topic_id"].ne(-1)].copy()
        frames.append(docs)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def validate_embedding_rows(frame: pd.DataFrame, embedding_rows: int) -> pd.DataFrame:
    docs = frame.copy()
    docs["embedding_row_index"] = pd.to_numeric(docs["embedding_row_index"], errors="coerce")
    missing = int(docs["embedding_row_index"].isna().sum())
    if missing:
        docs = docs.dropna(subset=["embedding_row_index"]).copy()
    docs["embedding_row_index"] = docs["embedding_row_index"].astype(int)
    valid = docs["embedding_row_index"].between(0, embedding_rows - 1)
    if not bool(valid.all()):
        docs = docs.loc[valid].copy()
    return docs.reset_index(drop=True)


def make_color_map(group_ids: list[str]) -> dict[str, str]:
    if not group_ids:
        return {}
    cmap = plt.get_cmap("turbo")
    if len(group_ids) == 1:
        return {group_ids[0]: "#2A7F62"}
    return {
        group_id: "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
        )
        for group_id, color in zip(group_ids, (cmap(value) for value in np.linspace(0.05, 0.95, len(group_ids))), strict=True)
    }


def fit_umap(matrix: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    reducer = UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
        transform_seed=args.seed,
        low_memory=True,
    )
    return np.asarray(reducer.fit_transform(matrix), dtype="float32")


def build_category_coordinates(
    macro_topic: str,
    docs: pd.DataFrame,
    groups: pd.DataFrame,
    embeddings: np.memmap,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    category_docs = docs.loc[docs["assigned_label"].astype(str).eq(macro_topic)].copy()
    if category_docs.empty:
        return pd.DataFrame(), pd.DataFrame()

    category_groups = groups.loc[groups["macro_topic"].eq(macro_topic)].copy()
    category_docs = category_docs.merge(
        category_groups[
            [
                "subgroup",
                "final_merge_group_id",
                "macro_topic_name",
                "group_label_refined",
                "group_summary",
            ]
        ],
        on=["subgroup", "final_merge_group_id"],
        how="left",
    )

    category_docs = validate_embedding_rows(category_docs, embeddings.shape[0])
    matrix = np.asarray(embeddings[category_docs["embedding_row_index"].to_numpy(dtype=int)], dtype="float32")
    coords = fit_umap(matrix, args)

    category_docs["umap_x"] = coords[:, 0]
    category_docs["umap_y"] = coords[:, 1]
    category_docs["macro_topic"] = macro_topic
    group_order = (
        category_docs["final_merge_group_id"]
        .value_counts()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    category_docs["source_color"] = category_docs["source"].map(SOURCE_COLORS).fillna("#6E6E6E")
    category_docs["source_marker"] = category_docs["source"].map(SOURCE_MARKERS).fillna("o")
    category_docs["group_label_refined"] = category_docs["group_label_refined"].fillna("")
    category_docs["group_summary"] = category_docs["group_summary"].fillna("")
    category_docs["plot_label"] = category_docs["final_merge_group_id"].astype(str)

    label_groups = set(group_order[: max(args.max_labels, 0)])
    centroids = (
        category_docs.groupby(["macro_topic", "source", "subgroup", "final_merge_group_id"], dropna=False)
        .agg(
            macro_topic_name=("macro_topic_name", "first"),
            group_label_refined=("group_label_refined", "first"),
            group_summary=("group_summary", "first"),
            point_count=("chunk_id", "size"),
            unique_document_count=("source_doc_id", "nunique"),
            centroid_x=("umap_x", "mean"),
            centroid_y=("umap_y", "mean"),
            source_color=("source_color", "first"),
            source_marker=("source_marker", "first"),
        )
        .reset_index()
    )
    centroids["label_on_figure"] = centroids["final_merge_group_id"].isin(label_groups)
    centroids = centroids.sort_values(["macro_topic", "point_count"], ascending=[True, False]).reset_index(drop=True)
    return category_docs, centroids


def render_category_figure(
    macro_topic: str,
    macro_topic_name: str,
    coordinates: pd.DataFrame,
    centroids: pd.DataFrame,
    output_root: Path,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(10.8, 8.0))

    for source in SOURCE_ORDER:
        source_docs = coordinates.loc[coordinates["source"].eq(source)]
        if source_docs.empty:
            continue
        ax.scatter(
            source_docs["umap_x"],
            source_docs["umap_y"],
            s=7,
            c=SOURCE_COLORS.get(source, "#6E6E6E"),
            marker=SOURCE_MARKERS.get(source, "o"),
            alpha=0.58,
            linewidths=0,
            rasterized=True,
        )

    label_rows = centroids.loc[centroids["label_on_figure"]].copy()
    for row in label_rows.itertuples(index=False):
        ax.text(
            float(row.centroid_x),
            float(row.centroid_y),
            str(row.final_merge_group_id),
            fontsize=7,
            color="#222222",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.78, "linewidth": 0.4},
        )

    source_handles = [
        Line2D(
            [0],
            [0],
            marker=SOURCE_MARKERS[source],
            color="none",
            markerfacecolor=SOURCE_COLORS.get(source, "#6E6E6E"),
            markeredgecolor="none",
            markersize=6,
            label=SOURCE_LABELS[source],
        )
        for source in SOURCE_ORDER
        if source in set(coordinates["source"].astype(str))
    ]
    ax.legend(handles=source_handles, title="Source", loc="upper right", frameon=True, framealpha=0.9)

    unique_docs = int(coordinates["source_doc_id"].nunique()) if "source_doc_id" in coordinates.columns else int(len(coordinates))
    title = f"{macro_topic}: {macro_topic_name}" if macro_topic_name else macro_topic
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        0.01,
        (
            f"Retained paper topic groups only. Points={len(coordinates):,} chunks; "
            f"unique docs={unique_docs:,}; groups={coordinates['final_merge_group_id'].nunique():,}. "
            "Colors=sources; labels=largest groups."
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.82, "pad": 4},
    )
    fig.tight_layout()

    png = output_root / f"{macro_topic}_document_topic_map.png"
    pdf = output_root / f"{macro_topic}_document_topic_map.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_outputs(
    output_root: Path,
    coordinates: pd.DataFrame,
    centroids: pd.DataFrame,
    category_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    coordinate_columns = [
        "macro_topic",
        "macro_topic_name",
        "source",
        "subgroup",
        "final_merge_group_id",
        "group_label_refined",
        "chunk_id",
        "source_doc_id",
        "document_id",
        "doc_id",
        "year",
        "micro_topic_id",
        "raw_micro_topic_id",
        "merged_micro_topic_id",
        "embedding_row_index",
        "umap_x",
        "umap_y",
        "source_color",
        "source_marker",
        "plot_label",
    ]
    existing_coordinate_columns = [column for column in coordinate_columns if column in coordinates.columns]
    coordinates[existing_coordinate_columns].to_csv(output_root / "document_topic_map_coordinates.csv", index=False)
    centroids.to_csv(output_root / "document_topic_map_group_centroids.csv", index=False)
    category_summary.to_csv(output_root / "document_topic_map_category_summary.csv", index=False)
    (output_root / "document_topic_map_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    nice_theme()
    args.output_root.mkdir(parents=True, exist_ok=True)

    macro_topic_names = load_macro_topic_names(args.catalog)
    requested_topics = [topic for topic in args.macro_topics if topic in MACRO_TOPIC_ORDER]
    if not requested_topics:
        raise ValueError("No valid macro topics requested.")

    retained_groups, stage12_selected_group_rows = load_retained_groups(args, macro_topic_names)
    retained_groups = retained_groups.loc[retained_groups["macro_topic"].isin(requested_topics)].copy()
    docs = load_documents_for_groups(args.model_root, retained_groups)
    embeddings, embedding_meta = load_embedding_memmap(args.embedding_file, args.embedding_meta)

    all_coordinates: list[pd.DataFrame] = []
    all_centroids: list[pd.DataFrame] = []
    figure_outputs: dict[str, dict[str, str]] = {}
    category_rows: list[dict[str, Any]] = []

    for macro_topic in requested_topics:
        category_coords, category_centroids = build_category_coordinates(
            macro_topic=macro_topic,
            docs=docs,
            groups=retained_groups,
            embeddings=embeddings,
            args=args,
        )
        if category_coords.empty:
            category_rows.append(
                {
                    "macro_topic": macro_topic,
                    "macro_topic_name": macro_topic_names.get(macro_topic, ""),
                    "point_count": 0,
                    "unique_document_count": 0,
                    "retained_group_count": 0,
                    "status": "empty",
                }
            )
            continue

        png, pdf = render_category_figure(
            macro_topic=macro_topic,
            macro_topic_name=macro_topic_names.get(macro_topic, ""),
            coordinates=category_coords,
            centroids=category_centroids,
            output_root=args.output_root,
        )
        figure_outputs[macro_topic] = {"png": str(png), "pdf": str(pdf)}
        all_coordinates.append(category_coords)
        all_centroids.append(category_centroids)
        category_rows.append(
            {
                "macro_topic": macro_topic,
                "macro_topic_name": macro_topic_names.get(macro_topic, ""),
                "point_count": int(len(category_coords)),
                "unique_document_count": int(category_coords["source_doc_id"].nunique()),
                "retained_group_count": int(category_coords["final_merge_group_id"].nunique()),
                "source_count": int(category_coords["source"].nunique()),
                "status": "completed",
            }
        )
        print(png)
        print(pdf)

    coordinates = pd.concat(all_coordinates, ignore_index=True, sort=False) if all_coordinates else pd.DataFrame()
    centroids = pd.concat(all_centroids, ignore_index=True, sort=False) if all_centroids else pd.DataFrame()
    category_summary = pd.DataFrame(category_rows)
    excluded_group_keys = sorted(load_excluded_group_keys(args.exclusion_decisions))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Static Fig. 5-style UMAP maps of retained paper topic document rows by macro category.",
        "inputs": {
            "model_root": str(args.model_root),
            "selected_topics": str(args.selected_topics),
            "group_metadata": str(args.group_metadata),
            "previous_review_decisions": str(args.previous_review_decisions),
            "exclusion_decisions": str(args.exclusion_decisions),
            "catalog": str(args.catalog),
            "embedding_file": str(args.embedding_file),
            "embedding_meta": str(args.embedding_meta),
        },
        "embedding_meta_payload": embedding_meta,
        "parameters": {
            "macro_topics": requested_topics,
            "umap_metric": "cosine",
            "umap_n_components": 2,
            "umap_n_neighbors": args.n_neighbors,
            "umap_min_dist": args.min_dist,
            "seed": args.seed,
            "max_labels": args.max_labels,
            "unit_of_analysis": "BERTopic document row/chunk; source_doc_id retained for document-level review.",
            "visual_encoding": {"color": "source", "marker": "source", "labels": "largest final_merge_group_id centroids"},
        },
        "excluded_group_count": len(excluded_group_keys),
        "previous_delete_group_count": len(load_previous_delete_group_keys(args.previous_review_decisions)),
        "excluded_group_keys": [f"{subgroup}::{group_id}" for subgroup, group_id in excluded_group_keys],
        "row_counts": {
            "coordinate_rows": int(len(coordinates)),
            "centroid_rows": int(len(centroids)),
            "category_rows": int(len(category_summary)),
            "retained_group_rows": int(len(retained_groups)),
            "stage12_selected_group_rows": int(stage12_selected_group_rows),
        },
        "outputs": {
            "coordinates_csv": str(args.output_root / "document_topic_map_coordinates.csv"),
            "group_centroids_csv": str(args.output_root / "document_topic_map_group_centroids.csv"),
            "category_summary_csv": str(args.output_root / "document_topic_map_category_summary.csv"),
            "figures": figure_outputs,
        },
    }
    write_outputs(args.output_root, coordinates, centroids, category_summary, manifest)
    print(args.output_root / "document_topic_map_manifest.json")


if __name__ == "__main__":
    main()
