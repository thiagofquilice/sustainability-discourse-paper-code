#!/usr/bin/env python3
"""Render topic-vector alignment maps for retained corporate-focus topic groups."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PIPELINE_ROOT = Path("/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline")
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
DEFAULT_PROFILE_EMBEDDINGS = (
    PIPELINE_ROOT
    / "outputs"
    / "microtopic_merge_first_review_multiaspect"
    / "microtopic_profile_embeddings.npy"
)
DEFAULT_PROFILE_METADATA = (
    PIPELINE_ROOT
    / "outputs"
    / "microtopic_merge_first_review_multiaspect"
    / "microtopic_profile_embedding_metadata.csv"
)
DEFAULT_MICROTOPIC_TO_GROUP = (
    PIPELINE_ROOT
    / "outputs"
    / "microtopic_merge_first_group_review_multiaspect"
    / "microtopic_to_merged_group.csv"
)
DEFAULT_EMBEDDING_MANIFEST = (
    PIPELINE_ROOT
    / "outputs"
    / "microtopic_merge_first_review_multiaspect"
    / "embedding_manifest.json"
)
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "outputs" / "paper_tables" / "corporate_focus_topic_vector_maps"

MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
SOURCE_ORDER = ["academic", "media", "corporate"]
TARGET_LINK_SOURCES = ["academic", "media"]
SOURCE_MARKERS = {"academic": "o", "media": "^", "corporate": "s"}
SOURCE_COLORS = {"academic": "#3569B8", "media": "#D8842A", "corporate": "#2C8C62"}
SOURCE_LABELS = {"academic": "Academic", "media": "Media", "corporate": "Corporate"}
SIMILARITY_THRESHOLDS = [0.70, 0.75, 0.80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-topics", type=Path, default=DEFAULT_SELECTED_TOPICS)
    parser.add_argument("--group-metadata", type=Path, default=DEFAULT_GROUP_METADATA)
    parser.add_argument("--previous-review-decisions", type=Path, default=DEFAULT_PREVIOUS_REVIEW_DECISIONS)
    parser.add_argument("--exclusion-decisions", type=Path, default=DEFAULT_EXCLUSION_DECISIONS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--profile-embeddings", type=Path, default=DEFAULT_PROFILE_EMBEDDINGS)
    parser.add_argument("--profile-metadata", type=Path, default=DEFAULT_PROFILE_METADATA)
    parser.add_argument("--microtopic-to-group", type=Path, default=DEFAULT_MICROTOPIC_TO_GROUP)
    parser.add_argument("--embedding-manifest", type=Path, default=DEFAULT_EMBEDDING_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--macro-topics", nargs="*", default=MACRO_TOPIC_ORDER)
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


def read_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


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

    removed_keys = load_excluded_group_keys(args.exclusion_decisions) | load_previous_delete_group_keys(
        args.previous_review_decisions
    )
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


def require_columns(frame: pd.DataFrame, columns: set[str], source_name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def normalize_vector(vector: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return np.asarray(vector / norm, dtype="float32"), norm


def load_microtopic_members(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(args.profile_embeddings, mmap_mode="r")
    metadata = pd.read_csv(args.profile_metadata)
    mapping = pd.read_csv(args.microtopic_to_group)

    require_columns(
        metadata,
        {
            "embedding_row_index",
            "subgroup",
            "source",
            "assigned_label",
            "microtopic_id",
            "document_count",
            "chunk_count",
        },
        "profile metadata",
    )
    require_columns(
        mapping,
        {"subgroup", "source", "assigned_label", "micro_topic_id", "final_merge_group_id"},
        "microtopic-to-group mapping",
    )

    metadata = metadata.copy()
    mapping = mapping.copy()
    metadata["macro_topic"] = metadata["assigned_label"].astype(str)
    mapping["macro_topic"] = mapping["assigned_label"].astype(str)
    metadata["subgroup"] = metadata["subgroup"].astype(str)
    metadata["source"] = metadata["source"].astype(str)
    mapping["subgroup"] = mapping["subgroup"].astype(str)
    mapping["source"] = mapping["source"].astype(str)
    mapping["final_merge_group_id"] = mapping["final_merge_group_id"].astype(str)
    metadata["micro_topic_id"] = pd.to_numeric(metadata["microtopic_id"], errors="raise").astype(int)
    mapping["micro_topic_id"] = pd.to_numeric(mapping["micro_topic_id"], errors="raise").astype(int)
    metadata["embedding_row_index"] = pd.to_numeric(metadata["embedding_row_index"], errors="raise").astype(int)
    metadata["document_count"] = pd.to_numeric(metadata["document_count"], errors="coerce").fillna(0).astype(float)
    metadata["chunk_count"] = pd.to_numeric(metadata["chunk_count"], errors="coerce").fillna(0).astype(float)

    if metadata["embedding_row_index"].min() < 0 or metadata["embedding_row_index"].max() >= embeddings.shape[0]:
        raise ValueError("Profile metadata contains embedding_row_index values outside embedding array bounds.")

    metadata_columns = [
        "embedding_row_index",
        "subgroup",
        "source",
        "macro_topic",
        "micro_topic_id",
        "document_count",
        "chunk_count",
    ]
    if "microtopic_label_clean" in metadata.columns:
        metadata_columns.append("microtopic_label_clean")

    members = mapping[
        ["subgroup", "source", "macro_topic", "micro_topic_id", "final_merge_group_id"]
    ].merge(
        metadata[metadata_columns],
        on=["subgroup", "source", "macro_topic", "micro_topic_id"],
        how="left",
        validate="one_to_one",
    )
    missing_embeddings = members["embedding_row_index"].isna()
    if missing_embeddings.any():
        sample = members.loc[missing_embeddings, ["subgroup", "micro_topic_id"]].head(10).to_dict("records")
        raise ValueError(f"Missing profile embeddings for mapped microtopics; sample={sample}")
    members["embedding_row_index"] = members["embedding_row_index"].astype(int)
    return embeddings, members


def build_group_vectors(
    retained_groups: pd.DataFrame,
    embeddings: np.ndarray,
    members: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    key_columns = ["macro_topic", "source", "subgroup", "final_merge_group_id"]
    retained = retained_groups[key_columns + ["macro_topic_name", "group_label_refined", "group_summary"]].drop_duplicates()
    retained_members = members.merge(retained, on=key_columns, how="inner")

    retained_keys = set(map(tuple, retained[key_columns].astype(str).to_numpy()))
    member_keys = set(map(tuple, retained_members[key_columns].astype(str).drop_duplicates().to_numpy()))
    missing_keys = sorted(retained_keys - member_keys)
    if missing_keys:
        preview = [dict(zip(key_columns, key)) for key in missing_keys[:10]]
        raise ValueError(f"Retained groups without microtopic profile vectors; sample={preview}")

    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    grouped = retained_members.sort_values(key_columns + ["micro_topic_id"]).groupby(key_columns, sort=True, dropna=False)
    for vector_row_index, (key, group) in enumerate(grouped):
        macro_topic, source, subgroup, final_merge_group_id = key
        embedding_rows = group["embedding_row_index"].to_numpy(dtype=int)
        member_vectors = np.asarray(embeddings[embedding_rows], dtype="float32")
        weights = group["chunk_count"].to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
        weighted_vector = np.average(member_vectors, axis=0, weights=weights)
        normalized_vector, norm_before = normalize_vector(weighted_vector)
        vectors.append(normalized_vector)

        records.append(
            {
                "vector_row_index": vector_row_index,
                "macro_topic": str(macro_topic),
                "macro_topic_name": clean_text(group["macro_topic_name"].iloc[0]),
                "source": str(source),
                "subgroup": str(subgroup),
                "final_merge_group_id": str(final_merge_group_id),
                "group_label_refined": clean_text(group["group_label_refined"].iloc[0]),
                "group_summary": clean_text(group["group_summary"].iloc[0]),
                "member_microtopic_count": int(group["micro_topic_id"].nunique()),
                "member_microtopic_ids": ";".join(map(str, sorted(group["micro_topic_id"].astype(int).tolist()))),
                "document_count": int(round(float(group["document_count"].sum()))),
                "chunk_count": int(round(float(group["chunk_count"].sum()))),
                "vector_norm_before_normalization": norm_before,
                "vector_l2_norm": float(np.linalg.norm(normalized_vector)),
                "source_color": SOURCE_COLORS.get(str(source), "#6E6E6E"),
                "source_marker": SOURCE_MARKERS.get(str(source), "o"),
                "plot_label": str(final_merge_group_id),
            }
        )

    group_frame = pd.DataFrame(records)
    vector_matrix = np.vstack(vectors).astype("float32") if vectors else np.empty((0, embeddings.shape[1]), dtype="float32")
    group_frame["source_order"] = group_frame["source"].map({source: idx for idx, source in enumerate(SOURCE_ORDER)}).fillna(99)
    group_frame["macro_order"] = group_frame["macro_topic"].map({topic: idx for idx, topic in enumerate(MACRO_TOPIC_ORDER)}).fillna(99)
    group_frame = group_frame.sort_values(["macro_order", "source_order", "subgroup", "final_merge_group_id"]).reset_index(drop=True)
    return group_frame, vector_matrix


def fit_cosine_mds(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.empty((0, 2), dtype="float32")
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype="float32")
    if matrix.shape[0] == 2:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype="float32")

    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    squared_distance = distance**2
    row_count = matrix.shape[0]
    centering = np.eye(row_count, dtype="float64") - (np.ones((row_count, row_count), dtype="float64") / row_count)
    gram = -0.5 * centering @ squared_distance.astype("float64") @ centering
    gram = (gram + gram.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    coordinates = np.zeros((row_count, 2), dtype="float64")
    for output_dimension, eigen_index in enumerate(order[:2]):
        eigenvalue = float(eigenvalues[eigen_index])
        if eigenvalue > 0:
            coordinates[:, output_dimension] = eigenvectors[:, eigen_index] * np.sqrt(eigenvalue)
    return coordinates.astype("float32")

def add_topic_coordinates(group_frame: pd.DataFrame, vector_matrix: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    coordinate_frames: list[pd.DataFrame] = []
    for macro_topic in args.macro_topics:
        category = group_frame.loc[group_frame["macro_topic"].eq(macro_topic)].copy()
        if category.empty:
            continue
        category_vectors = vector_matrix[category["vector_row_index"].to_numpy(dtype=int)]
        coords = fit_cosine_mds(category_vectors)
        category["projection_x"] = coords[:, 0]
        category["projection_y"] = coords[:, 1]
        category["quadrant_x"] = 0.0
        category["quadrant_y"] = 0.0
        category["plot_size"] = scale_plot_sizes(category["chunk_count"])
        coordinate_frames.append(category)
    if not coordinate_frames:
        return pd.DataFrame()
    coordinates = pd.concat(coordinate_frames, ignore_index=True, sort=False)
    if not np.isfinite(coordinates[["projection_x", "projection_y"]].to_numpy(dtype=float)).all():
        raise ValueError("Computed non-finite projection coordinates.")
    return coordinates


def scale_plot_sizes(values: pd.Series) -> np.ndarray:
    logged = np.log1p(pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float))
    if logged.size == 0:
        return np.asarray([], dtype=float)
    min_value = float(np.min(logged))
    max_value = float(np.max(logged))
    if np.isclose(max_value, min_value):
        return np.full(logged.shape, 130.0)
    scaled = (logged - min_value) / (max_value - min_value)
    return 70.0 + (scaled * 190.0)


def compute_nearest_pairs(coordinates: pd.DataFrame, vector_matrix: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for macro_topic, category in coordinates.groupby("macro_topic", sort=True):
        corporate = category.loc[category["source"].eq("corporate")].copy()
        if corporate.empty:
            continue
        corporate_indices = corporate["vector_row_index"].to_numpy(dtype=int)
        corporate_vectors = vector_matrix[corporate_indices]

        for corporate_row in corporate.itertuples(index=False):
            corporate_vector = vector_matrix[int(corporate_row.vector_row_index)]
            for target_source in TARGET_LINK_SOURCES:
                target = category.loc[category["source"].eq(target_source)].copy()
                if target.empty:
                    continue
                target_indices = target["vector_row_index"].to_numpy(dtype=int)
                target_vectors = vector_matrix[target_indices]
                similarities = target_vectors @ corporate_vector
                best_position = int(np.argmax(similarities))
                target_row = target.iloc[best_position]
                target_vector = vector_matrix[int(target_row["vector_row_index"])]
                reciprocal_position = int(np.argmax(corporate_vectors @ target_vector))
                reciprocal_vector_row = int(corporate_indices[reciprocal_position])

                records.append(
                    {
                        "macro_topic": macro_topic,
                        "macro_topic_name": clean_text(corporate_row.macro_topic_name),
                        "source_group_id": corporate_row.final_merge_group_id,
                        "source_subgroup": corporate_row.subgroup,
                        "source_source": "corporate",
                        "source_group_label_refined": clean_text(corporate_row.group_label_refined),
                        "source_document_count": int(corporate_row.document_count),
                        "source_chunk_count": int(corporate_row.chunk_count),
                        "target_source": target_source,
                        "target_group_id": str(target_row["final_merge_group_id"]),
                        "target_subgroup": str(target_row["subgroup"]),
                        "target_group_label_refined": clean_text(target_row["group_label_refined"]),
                        "target_document_count": int(target_row["document_count"]),
                        "target_chunk_count": int(target_row["chunk_count"]),
                        "cosine_similarity_1024d": float(similarities[best_position]),
                        "reciprocal_nearest": bool(reciprocal_vector_row == int(corporate_row.vector_row_index)),
                        "source_vector_row_index": int(corporate_row.vector_row_index),
                        "target_vector_row_index": int(target_row["vector_row_index"]),
                        "source_projection_x": float(corporate_row.projection_x),
                        "source_projection_y": float(corporate_row.projection_y),
                        "target_projection_x": float(target_row["projection_x"]),
                        "target_projection_y": float(target_row["projection_y"]),
                    }
                )
    return pd.DataFrame(records)


def summarize_alignment(coordinates: pd.DataFrame, nearest_pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        category = coordinates.loc[coordinates["macro_topic"].eq(macro_topic)]
        if category.empty:
            continue
        macro_topic_name = clean_text(category["macro_topic_name"].iloc[0])
        corporate_count = int(category["source"].eq("corporate").sum())
        for target_source in TARGET_LINK_SOURCES:
            target_count = int(category["source"].eq(target_source).sum())
            subset = nearest_pairs.loc[
                nearest_pairs["macro_topic"].eq(macro_topic) & nearest_pairs["target_source"].eq(target_source)
            ].copy()
            similarities = subset["cosine_similarity_1024d"] if not subset.empty else pd.Series(dtype=float)
            record: dict[str, Any] = {
                "macro_topic": macro_topic,
                "macro_topic_name": macro_topic_name,
                "source_pair": f"corporate->{target_source}",
                "source_group_count": corporate_count,
                "target_group_count": target_count,
                "nearest_pair_count": int(len(subset)),
                "nearest_pair_coverage": float(len(subset) / corporate_count) if corporate_count else np.nan,
                "reciprocal_pair_count": int(subset["reciprocal_nearest"].sum()) if not subset.empty else 0,
                "reciprocal_pair_rate": float(subset["reciprocal_nearest"].mean()) if not subset.empty else np.nan,
            }
            if similarities.empty:
                record.update(
                    {
                        "mean_nearest_cosine": np.nan,
                        "median_nearest_cosine": np.nan,
                        "min_nearest_cosine": np.nan,
                        "max_nearest_cosine": np.nan,
                        "p25_nearest_cosine": np.nan,
                        "p75_nearest_cosine": np.nan,
                    }
                )
                for threshold in SIMILARITY_THRESHOLDS:
                    record[f"share_ge_{str(threshold).replace('.', '_')}"] = np.nan
            else:
                record.update(
                    {
                        "mean_nearest_cosine": float(similarities.mean()),
                        "median_nearest_cosine": float(similarities.median()),
                        "min_nearest_cosine": float(similarities.min()),
                        "max_nearest_cosine": float(similarities.max()),
                        "p25_nearest_cosine": float(similarities.quantile(0.25)),
                        "p75_nearest_cosine": float(similarities.quantile(0.75)),
                    }
                )
                for threshold in SIMILARITY_THRESHOLDS:
                    record[f"share_ge_{str(threshold).replace('.', '_')}"] = float((similarities >= threshold).mean())
            records.append(record)
    return pd.DataFrame(records)


def link_alpha(cosine_similarity: float) -> float:
    if not np.isfinite(cosine_similarity):
        return 0.15
    return float(np.clip(0.18 + ((cosine_similarity - 0.55) / 0.35) * 0.62, 0.18, 0.80))


def link_width(cosine_similarity: float) -> float:
    if not np.isfinite(cosine_similarity):
        return 0.6
    return float(np.clip(0.7 + ((cosine_similarity - 0.55) / 0.35) * 1.3, 0.7, 2.0))


def render_category_figure(
    macro_topic: str,
    macro_topic_name: str,
    coordinates: pd.DataFrame,
    nearest_pairs: pd.DataFrame,
    output_root: Path,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(11.2, 8.2))

    quadrant_x = float(coordinates["quadrant_x"].iloc[0])
    quadrant_y = float(coordinates["quadrant_y"].iloc[0])
    ax.axvline(quadrant_x, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    ax.axhline(quadrant_y, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)

    category_pairs = nearest_pairs.loc[nearest_pairs["macro_topic"].eq(macro_topic)].copy()
    for row in category_pairs.itertuples(index=False):
        ax.plot(
            [row.source_projection_x, row.target_projection_x],
            [row.source_projection_y, row.target_projection_y],
            color=SOURCE_COLORS.get(row.target_source, "#777777"),
            alpha=link_alpha(float(row.cosine_similarity_1024d)),
            linewidth=link_width(float(row.cosine_similarity_1024d)),
            zorder=1,
        )

    for source in SOURCE_ORDER:
        source_rows = coordinates.loc[coordinates["source"].eq(source)]
        if source_rows.empty:
            continue
        ax.scatter(
            source_rows["projection_x"],
            source_rows["projection_y"],
            s=source_rows["plot_size"],
            c=SOURCE_COLORS.get(source, "#6E6E6E"),
            marker=SOURCE_MARKERS.get(source, "o"),
            alpha=0.90,
            linewidths=0.55,
            edgecolors="white",
            zorder=3,
        )

    for row in coordinates.itertuples(index=False):
        ax.annotate(
            str(row.final_merge_group_id),
            (float(row.projection_x), float(row.projection_y)),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.8,
            color="#222222",
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "#D0D0D0",
                "alpha": 0.68,
                "linewidth": 0.3,
            },
            zorder=4,
        )

    source_handles = [
        Line2D(
            [0],
            [0],
            marker=SOURCE_MARKERS[source],
            color="none",
            markerfacecolor=SOURCE_COLORS.get(source, "#6E6E6E"),
            markeredgecolor="none",
            markersize=7,
            label=SOURCE_LABELS[source],
        )
        for source in SOURCE_ORDER
        if source in set(coordinates["source"].astype(str))
    ]
    source_legend = ax.legend(handles=source_handles, title="Source", loc="upper right", frameon=True, framealpha=0.9)
    ax.add_artist(source_legend)

    link_handles = [
        Line2D([0], [0], color=SOURCE_COLORS[source], linewidth=1.7, label=f"Nearest {SOURCE_LABELS[source].lower()}")
        for source in TARGET_LINK_SOURCES
        if source in set(category_pairs["target_source"].astype(str))
    ]
    if link_handles:
        ax.legend(handles=link_handles, title="Corporate links", loc="upper left", frameon=True, framealpha=0.9)

    title = f"{macro_topic}: {macro_topic_name}" if macro_topic_name else macro_topic
    ax.set_title(title)
    ax.set_xlabel("Cosine MDS 1 (topic vectors)")
    ax.set_ylabel("Cosine MDS 2 (topic vectors)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        0.01,
        (
            f"Points={len(coordinates):,} retained topic groups. "
            "Vectors=chunk-weighted microtopic profile embeddings. "
            "MDS=cosine distances; links=nearest neighbors by 1024D cosine."
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.82, "pad": 4},
    )
    fig.tight_layout()

    png = output_root / f"{macro_topic}_topic_vector_map.png"
    pdf = output_root / f"{macro_topic}_topic_vector_map.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_outputs(
    output_root: Path,
    coordinates: pd.DataFrame,
    nearest_pairs: pd.DataFrame,
    alignment_summary: pd.DataFrame,
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
        "group_summary",
        "member_microtopic_count",
        "member_microtopic_ids",
        "document_count",
        "chunk_count",
        "vector_row_index",
        "vector_norm_before_normalization",
        "vector_l2_norm",
        "projection_x",
        "projection_y",
        "quadrant_x",
        "quadrant_y",
        "source_color",
        "source_marker",
        "plot_size",
        "plot_label",
    ]
    coordinates[coordinate_columns].to_csv(output_root / "topic_vector_coordinates.csv", index=False)
    nearest_pairs.to_csv(output_root / "topic_vector_nearest_cross_source_pairs.csv", index=False)
    alignment_summary.to_csv(output_root / "topic_vector_alignment_summary.csv", index=False)
    (output_root / "topic_vector_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def validate_outputs(
    coordinates: pd.DataFrame,
    retained_groups: pd.DataFrame,
    vector_matrix: np.ndarray,
    nearest_pairs: pd.DataFrame,
) -> None:
    retained_keys = set(
        map(
            tuple,
            retained_groups[["macro_topic", "source", "subgroup", "final_merge_group_id"]].astype(str).to_numpy(),
        )
    )
    coordinate_keys = set(
        map(tuple, coordinates[["macro_topic", "source", "subgroup", "final_merge_group_id"]].astype(str).to_numpy())
    )
    if retained_keys != coordinate_keys:
        missing = sorted(retained_keys - coordinate_keys)[:10]
        extra = sorted(coordinate_keys - retained_keys)[:10]
        raise ValueError(f"Coordinate group keys do not match retained groups; missing={missing}; extra={extra}")

    if not np.isfinite(coordinates[["projection_x", "projection_y"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite 2D coordinates detected.")
    if not np.isfinite(vector_matrix).all():
        raise ValueError("Non-finite vector values detected.")
    norms = np.linalg.norm(vector_matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Vector matrix rows are not L2-normalized.")
    if not nearest_pairs.empty:
        similarities = nearest_pairs["cosine_similarity_1024d"].to_numpy(dtype=float)
        if not np.isfinite(similarities).all():
            raise ValueError("Non-finite nearest-neighbor similarities detected.")
        if np.any(similarities < -1.0001) or np.any(similarities > 1.0001):
            raise ValueError("Cosine similarities outside [-1, 1] detected.")


def main() -> None:
    args = parse_args()
    nice_theme()
    args.output_root.mkdir(parents=True, exist_ok=True)

    macro_topic_names = load_macro_topic_names(args.catalog)
    requested_topics = [topic for topic in args.macro_topics if topic in MACRO_TOPIC_ORDER]
    if not requested_topics:
        raise ValueError("No valid macro topics requested.")
    args.macro_topics = requested_topics

    retained_groups, stage12_selected_group_rows = load_retained_groups(args, macro_topic_names)
    retained_groups = retained_groups.loc[retained_groups["macro_topic"].isin(requested_topics)].copy()
    embeddings, members = load_microtopic_members(args)
    group_frame, vector_matrix = build_group_vectors(retained_groups, embeddings, members)
    coordinates = add_topic_coordinates(group_frame, vector_matrix, args)
    nearest_pairs = compute_nearest_pairs(coordinates, vector_matrix)
    alignment_summary = summarize_alignment(coordinates, nearest_pairs)
    validate_outputs(coordinates, retained_groups, vector_matrix, nearest_pairs)

    figure_outputs: dict[str, dict[str, str]] = {}
    category_rows: list[dict[str, Any]] = []
    for macro_topic in requested_topics:
        category = coordinates.loc[coordinates["macro_topic"].eq(macro_topic)].copy()
        if category.empty:
            category_rows.append(
                {
                    "macro_topic": macro_topic,
                    "macro_topic_name": macro_topic_names.get(macro_topic, ""),
                    "retained_group_count": 0,
                    "status": "empty",
                }
            )
            continue
        category_pairs = nearest_pairs.loc[nearest_pairs["macro_topic"].eq(macro_topic)].copy()
        png, pdf = render_category_figure(
            macro_topic=macro_topic,
            macro_topic_name=macro_topic_names.get(macro_topic, ""),
            coordinates=category,
            nearest_pairs=category_pairs,
            output_root=args.output_root,
        )
        figure_outputs[macro_topic] = {"png": str(png), "pdf": str(pdf)}
        category_rows.append(
            {
                "macro_topic": macro_topic,
                "macro_topic_name": macro_topic_names.get(macro_topic, ""),
                "retained_group_count": int(len(category)),
                "corporate_group_count": int(category["source"].eq("corporate").sum()),
                "academic_group_count": int(category["source"].eq("academic").sum()),
                "media_group_count": int(category["source"].eq("media").sum()),
                "nearest_link_count": int(len(category_pairs)),
                "quadrant_x": float(category["quadrant_x"].iloc[0]),
                "quadrant_y": float(category["quadrant_y"].iloc[0]),
                "status": "completed",
            }
        )
        print(png)
        print(pdf)

    excluded_group_keys = sorted(load_excluded_group_keys(args.exclusion_decisions))
    previous_delete_group_keys = sorted(load_previous_delete_group_keys(args.previous_review_decisions))
    embedding_manifest = read_json_or_empty(args.embedding_manifest)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Static 2D cosine MDS maps of retained paper topic-group vectors by macro category.",
        "inputs": {
            "selected_topics": str(args.selected_topics),
            "group_metadata": str(args.group_metadata),
            "previous_review_decisions": str(args.previous_review_decisions),
            "exclusion_decisions": str(args.exclusion_decisions),
            "catalog": str(args.catalog),
            "profile_embeddings": str(args.profile_embeddings),
            "profile_metadata": str(args.profile_metadata),
            "microtopic_to_group": str(args.microtopic_to_group),
            "embedding_manifest": str(args.embedding_manifest),
        },
        "embedding_manifest_payload": embedding_manifest,
        "parameters": {
            "macro_topics": requested_topics,
            "projection_method": "classical_mds_pcoa",
            "projection_distance": "cosine distance (1 - cosine similarity) from normalized 1024D topic-group vectors",
            "projection_dimensions": 2,
            "unit_of_analysis": "retained final_merge_group_id topic group",
            "vector_basis": "microtopic profile embeddings from profile_text",
            "vector_aggregation": "chunk_count-weighted mean of member microtopic profile embeddings, then L2 normalization",
            "alignment_evidence": "nearest corporate-to-external neighbors by cosine similarity in original 1024D normalized vector space",
            "quadrant_reference": "cosine MDS origin at x=0/y=0; visual reference only",
            "visual_encoding": {
                "point_color": "source",
                "point_marker": "source",
                "point_size": "log-scaled chunk_count",
                "point_label": "final_merge_group_id",
                "line": "nearest corporate-to-academic and corporate-to-media neighbor links",
                "line_alpha": "cosine_similarity_1024d",
            },
        },
        "excluded_group_count": len(excluded_group_keys),
        "previous_delete_group_count": len(previous_delete_group_keys),
        "excluded_group_keys": [f"{subgroup}::{group_id}" for subgroup, group_id in excluded_group_keys],
        "previous_delete_group_keys": [f"{subgroup}::{group_id}" for subgroup, group_id in previous_delete_group_keys],
        "row_counts": {
            "coordinate_rows": int(len(coordinates)),
            "nearest_pair_rows": int(len(nearest_pairs)),
            "alignment_summary_rows": int(len(alignment_summary)),
            "retained_group_rows": int(len(retained_groups)),
            "stage12_selected_group_rows": int(stage12_selected_group_rows),
            "microtopic_member_rows": int(len(members)),
            "vector_dimensions": int(vector_matrix.shape[1]) if vector_matrix.size else 0,
        },
        "category_summary": category_rows,
        "outputs": {
            "coordinates_csv": str(args.output_root / "topic_vector_coordinates.csv"),
            "nearest_cross_source_pairs_csv": str(args.output_root / "topic_vector_nearest_cross_source_pairs.csv"),
            "alignment_summary_csv": str(args.output_root / "topic_vector_alignment_summary.csv"),
            "figures": figure_outputs,
        },
    }
    write_outputs(args.output_root, coordinates, nearest_pairs, alignment_summary, manifest)
    print(args.output_root / "topic_vector_manifest.json")


if __name__ == "__main__":
    main()
