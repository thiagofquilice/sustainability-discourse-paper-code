#!/usr/bin/env python3
"""Validate provisional six-domain assignments with Gemma 4 in a GPU runtime."""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = WORKFLOW_ROOT / "outputs" / "cosine" / "best_only_positive.parquet"
DEFAULT_CATALOG = WORKFLOW_ROOT / "catalog" / "six_topic_discourse_catalog.csv"
DEFAULT_OUTPUT_DIR = WORKFLOW_ROOT / "outputs" / "validation"
DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"

PROMPT_TEMPLATE = """You are validating one provisional environmental-domain assignment.

Answer yes only when the text substantively discusses at least one listed domain element or a close conceptual equivalent. Answer no for generic references to ESG, sustainability, environmental issues, risk, finance, or investment that do not substantively match the assigned domain.

Return exactly one word, either yes or no.

Assigned domain code: {assigned_label}
Assigned domain name: {topic_name}
Domain definition: {topic_definition}
Domain elements:
{assigned_label_text}

Text:
{text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def print_log(message: str) -> None:
    print(f"[{now_utc_iso()}] {message}", flush=True)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def normalize_binary_response(raw_text: Any) -> str:
    words = re.findall(r"[a-z]+", str(raw_text).strip().lower())
    for word in words:
        if word in {"yes", "no"}:
            return word
    return "invalid"


def load_and_enrich_input(input_path: Path, catalog_path: Path) -> pd.DataFrame:
    frame = read_table(input_path)
    if frame.empty:
        raise SystemExit("Validation input contains no rows.")
    required = {"chunk_id", "text", "assigned_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Validation input is missing required columns: {missing}")

    catalog = pd.read_csv(catalog_path)
    catalog = catalog[
        [
            "topic_code",
            "topic_name",
            "topic_definition",
            "topic_elements_bullets",
        ]
    ].rename(
        columns={
            "topic_code": "assigned_label",
            "topic_elements_bullets": "assigned_label_text",
        }
    )
    refresh_columns = ["topic_name", "topic_definition", "assigned_label_text"]
    base_columns = [column for column in frame.columns if column not in refresh_columns]
    enriched = frame[base_columns].merge(catalog, on="assigned_label", how="left")
    if enriched[refresh_columns].isna().any(axis=None):
        unknown = sorted(enriched.loc[enriched["topic_name"].isna(), "assigned_label"].unique())
        raise SystemExit(f"Validation input contains unknown assigned labels: {unknown}")
    enriched["chunk_id"] = enriched["chunk_id"].astype(str)
    if enriched["chunk_id"].duplicated().any():
        raise SystemExit("Validation input contains duplicated chunk_id values.")
    return enriched.reset_index(drop=True)


def build_prompt(row: pd.Series) -> str:
    return PROMPT_TEMPLATE.format(
        assigned_label=str(row["assigned_label"]),
        topic_name=str(row["topic_name"]),
        topic_definition=str(row["topic_definition"]),
        assigned_label_text=str(row["assigned_label_text"]),
        text=str(row["text"]),
    )


def load_model_and_processor(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Use a Colab GPU runtime.")

    model_kwargs: dict[str, Any] = {"device_map": "auto", "dtype": "auto"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()
    hardware = {
        "python_version": platform.python_version(),
        "transformers_version": transformers.__version__,
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": int(torch.cuda.device_count()),
    }
    return model, processor, hardware


def generate_batch(model, processor, rows: list[pd.Series], args: argparse.Namespace) -> list[str]:
    rendered_prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": build_prompt(row)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    inputs = processor(
        text=rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    return processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)


def write_checkpoint(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_csv = output_dir / "validation_output.csv"
    output_parquet = output_dir / "validation_output.parquet"
    frame.to_csv(output_csv, index=False)
    parquet_status = "written"
    try:
        frame.to_parquet(output_parquet, index=False)
    except Exception as exc:  # noqa: BLE001
        parquet_status = f"failed: {exc}"
    return {"csv": str(output_csv), "parquet": str(output_parquet), "parquet_status": parquet_status}


def main() -> None:
    args = parse_args()
    if args.max_new_tokens != 1:
        raise SystemExit("The thesis protocol requires --max-new-tokens 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_and_enrich_input(args.input, args.catalog)
    if args.max_rows is not None:
        frame = frame.head(args.max_rows).copy()

    output_csv = args.output_dir / "validation_output.csv"
    completed = pd.DataFrame()
    completed_ids: set[str] = set()
    if args.resume and output_csv.exists():
        completed = pd.read_csv(output_csv, low_memory=False)
        if {"chunk_id", "v_gemma"}.issubset(completed.columns):
            completed["chunk_id"] = completed["chunk_id"].astype(str)
            valid_completed = completed["v_gemma"].astype(str).str.lower().isin(
                {"yes", "no", "invalid"}
            )
            completed = completed.loc[valid_completed].copy()
            completed_ids = set(completed["chunk_id"])
    pending = frame.loc[~frame["chunk_id"].astype(str).isin(completed_ids)].copy()

    new_rows: list[dict[str, Any]] = []
    total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    if pending.empty:
        hardware = {"model_loaded": False, "reason": "all rows were already completed"}
    else:
        model, processor, hardware = load_model_and_processor(args)
        for batch_index, start in enumerate(range(0, len(pending), args.batch_size), start=1):
            batch = pending.iloc[start : start + args.batch_size]
            raw_outputs = generate_batch(
                model,
                processor,
                [row for _, row in batch.iterrows()],
                args,
            )
            for (_, row), raw_output in zip(batch.iterrows(), raw_outputs):
                record = row.to_dict()
                record["v_gemma"] = normalize_binary_response(raw_output)
                record["v_gemma_raw"] = str(raw_output).strip()
                new_rows.append(record)

            if batch_index % args.save_every == 0 or batch_index == total_batches:
                current = pd.concat(
                    [completed, pd.DataFrame(new_rows)],
                    ignore_index=True,
                    sort=False,
                )
                current = current.drop_duplicates(subset=["chunk_id"], keep="last")
                write_checkpoint(current, args.output_dir)
                print_log(f"saved batch {batch_index}/{total_batches} rows={len(current)}")

    final = pd.concat([completed, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    final = final.drop_duplicates(subset=["chunk_id"], keep="last")
    input_order = {chunk_id: index for index, chunk_id in enumerate(frame["chunk_id"])}
    final["_input_order"] = final["chunk_id"].map(input_order)
    final = final.sort_values("_input_order").drop(columns="_input_order").reset_index(drop=True)
    outputs = write_checkpoint(final, args.output_dir)

    manifest = {
        "input": str(args.input),
        "catalog": str(args.catalog),
        "model": args.model_name,
        "binary_output": True,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "load_in_4bit": args.load_in_4bit,
        "row_count": int(len(final)),
        "yes_count": int((final["v_gemma"] == "yes").sum()),
        "no_count": int((final["v_gemma"] == "no").sum()),
        "invalid_count": int((final["v_gemma"] == "invalid").sum()),
        "hardware": hardware,
        "outputs": outputs,
    }
    (args.output_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
