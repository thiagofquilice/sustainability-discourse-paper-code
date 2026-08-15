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

The corpus needs the seven columns documented in the README. Row order is the link between the corpus and the float32 embedding memmap.

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

The resulting `best_only_positive.parquet` contains only assignments that satisfy both the `0.65` similarity threshold and the `0.02` score margin.

## Stage 3 Gemma relevance validation

Run this stage in a GPU environment with access to `google/gemma-4-E4B-it`.

```bash
python pipeline/02_topic_modeling/run_hf_gemma_domain_validation_colab.py \
  --input outputs/cosine/best_only_positive.parquet \
  --output-dir outputs/validation \
  --model-name google/gemma-4-E4B-it \
  --max-new-tokens 1
```

The script records the raw response and a normalized `v_gemma` value. Invalid responses remain visible and are excluded from the validated `yes` corpus.

## Stage 4 validated modeling corpus

```bash
python pipeline/02_topic_modeling/materialize_adjusted_full_validation_output.py \
  --full-output outputs/validation/validation_output.csv \
  --best-only-positive outputs/cosine/best_only_positive.parquet \
  --output-dir outputs/validation/adjusted
```

The optional arguments `--remap-file` and `--changed-gemma` reproduce the study specific T2 secondary recovery when those two original intermediate files are available. Both arguments must be supplied together. The generic route keeps the validated assignments without that additional adjustment.

## Stage 5 source specific BERTopic models

```bash
python pipeline/02_topic_modeling/run_6topic_micro_unsupervised.py \
  --input outputs/validation/adjusted/adjusted_full_yes.csv \
  --embedding-file data/external/filtered_embeddings.f32 \
  --embedding-meta data/external/filtered_embeddings.meta.json \
  --output-root outputs/bertopic_micro_unsupervised_multiaspect
```

The script fits one model for every nonempty source and domain subgroup. Topic `-1` remains an outlier and does not enter later review.

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

The matching script writes the full pairwise similarity table. Corporate selection uses every direct academic to corporate or media to corporate relation at or above `0.65`. Mutual nearest neighbor tables are diagnostics and do not determine inclusion.

## Stage 8 corporate centered review

```bash
python pipeline/04_classification_and_review/build_corporate_focus_review.py \
  --merged-micro-root outputs/bertopic_micro_merged_multiaspect_reviewed \
  --pair-root outputs/microtopic_cross_source_pairs_corporate_focus \
  --output-root outputs/corporate_focus_review \
  --similarity-threshold 0.65
```

This command creates the threshold based candidate layer for the first human review. The hybrid workbook is created only after temporal synthesis because it requires the completed Stage 2 narratives.

After the reviewed merge workbook exists, the convenience wrapper can repeat merge materialization, cross source comparison, corporate candidate construction, and Stage 1 and 2 input preparation.

```bash
python pipeline/04_classification_and_review/run_corporate_focus_stage12_prep.py \
  --similarity-threshold 0.65
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

The two Gemma runners inside `pipeline/03_topic_description_and_interpretation` execute annual summaries and temporal evolution synthesis. They require model access and a GPU. The Drive bundle builder packages those runners and their inputs for Colab.

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

This workbook supports the final human audit of corporate relevance, paired external issues, relevant unpaired issues, and exclusions while displaying the temporal narratives on both sides of each comparison.

## Verification without research models

```bash
python -m compileall -q pipeline
for script in $(find pipeline -type f -name '*.py' ! -path 'pipeline/shared/*' | sort); do
  python "$script" --help >/dev/null
done
```

This structural check does not replace the integration test with corpus embeddings, BERTopic dependencies, reviewed workbooks, and the Gemma model.
