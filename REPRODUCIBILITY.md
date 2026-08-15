# Reproducibility Notes

This guide separates executable stages from the two human review boundaries. Commands are shown from the repository root after environment activation.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-full.txt
python -m spacy download en_core_web_sm
cp config/paper_6topic_pipeline_config.example.json config/paper_6topic_pipeline_config.json
```

Edit the copied configuration so the corpus and embedding paths refer to local files.

## Stage 1 corpus embeddings

The corpus needs the seven columns documented in the README. Row order is the link between the corpus and the float32 embedding memmap. The script below creates normalized embeddings with the model and dimensions required by the assignment stage.

```bash
python pipeline/01_data_preparation/embed_corpus.py \
  --input data/external/filtered_corpus.parquet \
  --embedding-file data/external/filtered_embeddings.f32 \
  --embedding-meta data/external/filtered_embeddings.meta.json \
  --model-name BAAI/bge-large-en-v1.5
```

## Stage 2 provisional six domain assignment

```bash
python pipeline/02_topic_modeling/run_6topic_discourse_cosine.py \
  --config config/paper_6topic_pipeline_config.json \
  --output-root outputs/cosine
```

`best_only_positive.parquet` contains assignments whose best domain score is at least `0.65`. The `score_gap`/`score_margin=0.02` field is retained as a diagnostic. Setting `assignment.apply_score_margin_to_positive_mask=true` additionally requires that margin and can be used as a sensitivity analysis.

## Stage 3 Gemma relevance validation

Run this stage in a GPU environment with access to `google/gemma-4-E4B-it`.

```bash
python pipeline/02_topic_modeling/run_hf_gemma_domain_validation_colab.py \
  --input-path outputs/cosine/best_only_positive.parquet \
  --output-path outputs/validation/validation_output.csv \
  --model-name google/gemma-4-E4B-it \
  --batch-size 256 \
  --save-every 2048 \
  --max-new-tokens 1
```

The script uses `AutoModelForCausalLM`, 4-bit quantization, deterministic decoding, and a binary prompt. It records `v_gemma` and an error log. Malformed responses are written to the error log rather than silently converted.

For a CPU smoke test on a few rows, keep the same model and prompt but request CPU explicitly and use batch size 1. This disables BitsAndBytes quantization and is not intended for a full validation run.

```bash
python pipeline/02_topic_modeling/run_hf_gemma_domain_validation_colab.py \
  --input-path outputs/cosine/best_only_positive.parquet \
  --output-path outputs/validation_smoke/validation_output.csv \
  --model-name google/gemma-4-E4B-it \
  --device cpu \
  --batch-size 1 \
  --max-rows 6 \
  --max-new-tokens 1
```

## Stage 4 validated modeling corpus

```bash
python pipeline/02_topic_modeling/materialize_adjusted_full_validation_output.py \
  --full-output outputs/validation/validation_output.csv \
  --best-only-positive outputs/cosine/best_only_positive.parquet \
  --remap-file outputs/t2_secondary_recovery_cosine_round/old_t2_under_secondary_recovery_variant.parquet \
  --changed-gemma outputs/t2_secondary_recovery_cosine_round/gemma_local_full/validation_output.csv \
  --output-dir outputs/validation/adjusted
```

The Chapter 4 route applies the study-specific T2 secondary recovery by default and records its reconciliation in the materialization manifest. For a new corpus that has not passed through that review, run the same command with `--adjustment-mode generic_no_t2_recovery`. In that mode the remap and secondary validation files are not required.

## Stage 5 source specific BERTopic models

```bash
python pipeline/02_topic_modeling/run_6topic_micro_unsupervised.py \
  --input outputs/validation/adjusted/adjusted_full_yes.csv \
  --embedding-file data/external/filtered_embeddings.f32 \
  --embedding-meta data/external/filtered_embeddings.meta.json \
  --output-root outputs/bertopic_micro_unsupervised_multiaspect
```

The script fits one model for every nonempty source and domain subgroup. Topic `-1` remains an outlier and does not enter later review.
Each completed subgroup writes `document_topics`, `topic_info`, topic-aspect files, and a saved BERTopic model directory. The summary manifest is written both to `outputs/bertopic_micro_unsupervised_multiaspect/summary/manifest.json` and to `outputs/bertopic_micro_unsupervised_multiaspect/manifest.json`.

## Stage 6 human merge review

Create the workbook.

```bash
python pipeline/02_topic_modeling/build_microtopic_merge_review_workbook.py \
  --micro-root outputs/bertopic_micro_unsupervised_multiaspect \
  --output-root outputs/microtopic_merge_first_group_review_multiaspect
```

Review the workbook. Move topics that form a possible merge from `singletons` to `group_members`, assign a shared `proposed_group_id`, and add the corresponding row to `group_summary`. Record `accept`, `reject`, or `split`. For split decisions, fill `manual_split_group_id` for every member.

Materialize the reviewed decisions and build the merged root.

```bash
python pipeline/02_topic_modeling/materialize_microtopic_group_review_mapping.py \
  --output-root outputs/microtopic_merge_first_group_review_multiaspect \
  --workbook outputs/microtopic_merge_first_group_review_multiaspect/microtopic_merge_first_group_review_multiaspect.xlsx

python pipeline/02_topic_modeling/build_merged_microtopic_root.py \
  --micro-root outputs/bertopic_micro_unsupervised_multiaspect \
  --mapping-csv outputs/microtopic_merge_first_group_review_multiaspect/microtopic_to_merged_group.csv \
  --output-root outputs/bertopic_micro_merged_multiaspect_reviewed
```

## Stage 7 cross source comparison

```bash
python pipeline/05_cross_source_comparison/build_cross_source_microtopic_profiles.py \
  --micro-root outputs/bertopic_micro_merged_multiaspect_reviewed \
  --output-root outputs/microtopic_cross_source_pairs_corporate_focus

python pipeline/05_cross_source_comparison/embed_cross_source_microtopic_profiles.py \
  --output-root outputs/microtopic_cross_source_pairs_corporate_focus \
  --model-name BAAI/bge-large-en-v1.5

python pipeline/05_cross_source_comparison/match_cross_source_microtopics.py \
  --output-root outputs/microtopic_cross_source_pairs_corporate_focus
```

The matching script writes the full pairwise similarity table. Corporate selection first uses every direct academic-to-corporate or media-to-corporate relation at or above `0.60`. Mutual nearest neighbor tables are diagnostics and do not determine inclusion.

## Stage 8 corporate centered review

```bash
python pipeline/04_classification_and_review/build_corporate_focus_review.py \
  --merged-micro-root outputs/bertopic_micro_merged_multiaspect_reviewed \
  --pair-root outputs/microtopic_cross_source_pairs_corporate_focus \
  --output-root outputs/corporate_focus_review \
  --similarity-threshold 0.60 \
  --primary-threshold 0.65
```

This command creates the threshold-based candidate layer for the first human review. The `--primary-threshold 0.65` default keeps the main reference threshold explicit. Selection is not a fully automatic 0.65-only rule because the review band and manual decisions are part of the method. The hybrid workbook is created only after temporal synthesis because it requires the completed Stage 2 narratives.

After the reviewed merge workbook exists, the convenience wrapper can repeat merge materialization, cross source comparison, corporate candidate construction, and Stage 1 and 2 input preparation.

```bash
python pipeline/04_classification_and_review/run_corporate_focus_stage12_prep.py \
  --similarity-threshold 0.60 \
  --primary-threshold 0.65
```

## Stage 9 temporal synthesis

Prepare evidence and the upload package.

```bash
python pipeline/03_topic_description_and_interpretation/build_corporate_focus_stage12_inputs.py \
  --micro-root outputs/bertopic_micro_merged_multiaspect_reviewed \
  --review-root outputs/corporate_focus_review \
  --output-root outputs/corporate_focus_stage12_input

python pipeline/03_topic_description_and_interpretation/build_corporate_focus_colab_package.py \
  --input-root outputs/corporate_focus_stage12_input

python pipeline/03_topic_description_and_interpretation/build_corporate_focus_drive_colab_bundle.py \
  --source-input-root outputs/corporate_focus_stage12_input \
  --bundle-root outputs/corporate_focus_stage12_colab_drive \
  --zip-path outputs/corporate_focus_stage12_colab_drive.zip
```

The two Gemma runners inside `pipeline/03_topic_description_and_interpretation` execute annual summaries and temporal evolution synthesis with `AutoModelForCausalLM`. Their defaults are Stage 1 `batch_size=6`, `save_every=20`, `max_new_tokens=320`, and Stage 2 `max_new_tokens=650`, `max_attempts=4`. They require model access and a GPU. The Drive bundle builder packages those runners and their inputs for Colab.

For CPU smoke tests only, pass `--device cpu` and a small `--max-rows` value to the annual runner, and `--device cpu --max-rows 1` to the evolution runner. CPU mode preserves the same prompts and `AutoModelForCausalLM` loader but disables BitsAndBytes quantization.

## Stage 10 post synthesis hybrid review

After Stage 2 has produced `micro_topic_evolution_narratives.csv` inside the bundle's `colab_outputs/phase_02_evolution_summaries` directory, build the hybrid review workbook.

```bash
python pipeline/04_classification_and_review/build_corporate_focus_hybrid_review_workbook.py \
  --bundle-root outputs/corporate_focus_stage12_colab_drive \
  --review-root outputs/corporate_focus_review \
  --stage12-input-root outputs/corporate_focus_stage12_input \
  --pair-root outputs/microtopic_cross_source_pairs_corporate_focus \
  --merged-micro-root outputs/bertopic_micro_merged_multiaspect_reviewed
```

This workbook supports the final human review of corporate relevance, paired external issues, relevant unpaired issues, and exclusions while displaying the temporal narratives on both sides of each comparison.

## ODS/SDG metadata

The SDG crosswalks are defined in the six-topic catalog and copied to assignment outputs as metadata. Six-domain embeddings are computed from descriptor expressions and subanchors, and corpus embeddings are computed from the corpus text. The SDG crosswalk strings are not embedded for assignment.


## Dependency compatibility

`requirements-full.txt` specifies a compatible execution environment. Generated manifests record relevant runtime and model information so readers can document the environment used for their own run.

## Verification without research models

```bash
python -m compileall -q pipeline
for script in $(find pipeline -type f -name '*.py' ! -path 'pipeline/shared/*' | sort); do
  python "$script" --help >/dev/null
done
```

This structural check does not replace the integration test with corpus embeddings, BERTopic dependencies, reviewed workbooks, and the Gemma model.
