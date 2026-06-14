# Sustainability Discourse Paper Code

Minimal code supplement for Chapter 4 of the thesis/paper.

This repository keeps only the scripts needed to understand the computational route used in the chapter:

1. classify texts into six environmental discourse domains;
2. run BERTopic separately by source and domain;
3. prepare selected microtopics for Colab/Gemma synthesis;
4. review and select microtopics around corporate anchors;
5. compare selected microtopics across sources.

No empirical data are included. The repository does not include corpus files, chunks, embeddings, generated CSV/Parquet/XLSX files, BERTopic models, filled workbooks, Colab input/output bundles, rendered figures, or final paper tables.

## Scripts, Inputs, and Outputs

The route below shows what each step needs and what it produces. Inputs and outputs are expected to exist locally, outside version control.

| Step | Purpose | Inputs | Scripts | Outputs |
|---|---|---|---|---|
| 1. Define six domains | Create the environmental-domain catalog used for classification. | No external input. The catalog is defined in code/static catalog files. | `pipeline/02_topic_modeling/six_topic_discourse_catalog.py` | Six-domain catalog in `catalog/`, used by later assignment scripts. |
| 2. Assign texts to domains | Link chunks/documents to the six domains using embeddings and cosine similarity. | Local prepared corpus, document/chunk metadata, local embeddings, and local config. | `pipeline/02_topic_modeling/run_6topic_discourse_cosine.py` | Local domain-assignment outputs, including labeled corpus tables. |
| 3. Apply validation adjustments | Apply validation and manual adjustment decisions before topic modeling. | Local validation outputs and local adjustment/review decisions. | `pipeline/02_topic_modeling/materialize_adjusted_full_validation_output.py` | Adjusted full assignment table and filtered modeling corpus. |
| 4. Run BERTopic by source and domain | Fit separate BERTopic models for each source-domain subgroup. | Adjusted modeling corpus and local embeddings. | `pipeline/02_topic_modeling/run_6topic_micro_unsupervised.py` | Local BERTopic microtopic outputs by source and domain. |
| 5. Merge reviewed microtopics | Convert review decisions into a merged microtopic structure. | Raw BERTopic outputs and local microtopic review decisions/workbook. | `pipeline/02_topic_modeling/materialize_microtopic_group_review_mapping.py`; `pipeline/02_topic_modeling/build_merged_microtopic_root.py` | Reviewed mapping files and merged microtopic root. |
| 6. Compare microtopics across sources | Build profile texts, embed them, and match similar microtopics across source pairs. | Merged microtopic root, profile text inputs, and embedding configuration. | `pipeline/05_cross_source_comparison/build_cross_source_microtopic_profiles.py`; `pipeline/05_cross_source_comparison/embed_cross_source_microtopic_profiles.py`; `pipeline/05_cross_source_comparison/match_cross_source_microtopics.py` | Local profile embeddings and cross-source candidate match tables. |
| 7. Select corporate-focused topics | Build and review the corporate-centered subset of microtopics. | Merged microtopics, cross-source matches, corporate-anchor review decisions, and optional overrides. | `pipeline/04_classification_and_review/build_corporate_focus_review.py`; `pipeline/04_classification_and_review/build_corporate_focus_hybrid_review_workbook.py`; `pipeline/04_classification_and_review/build_corporate_focus_override_packages.py`; `pipeline/04_classification_and_review/run_corporate_focus_stage12_prep.py` | Corporate-focused review files, review workbook, override packages, and Stage 1/2 preparation outputs. |
| 8. Run Colab/Gemma temporal synthesis | Prepare Colab inputs and run annual plus temporal LLM synthesis. | Selected corporate-focused microtopics, year evidence, Colab/Gemma access, and downloaded Colab outputs for restore. | `pipeline/03_topic_description_and_interpretation/build_corporate_focus_stage12_inputs.py`; `pipeline/03_topic_description_and_interpretation/build_corporate_focus_colab_package.py`; `pipeline/03_topic_description_and_interpretation/build_corporate_focus_drive_colab_bundle.py`; `pipeline/03_topic_description_and_interpretation/run_hf_gemma_micro_topic_year_summaries_colab.py`; `pipeline/03_topic_description_and_interpretation/run_hf_gemma_micro_topic_evolution_synthesis_colab.py`; `pipeline/03_topic_description_and_interpretation/restore_corporate_focus_colab_outputs.py` | Colab input bundles, annual summaries, temporal evolution narratives, and restored local output folders. |
| Shared helpers | Provide common path, config, text, workbook, and workflow utilities used by the scripts above. | Local path configuration and the files passed to each step script. | `pipeline/shared/*.py` | Helper functions only; no direct pipeline output. |

## Local Setup

Copy the example config and point it to local external files:

```bash
cp config/paper_6topic_pipeline_config.example.json config/paper_6topic_pipeline_config.json
```

Run scripts from the repository root with shared helpers on `PYTHONPATH`:

```bash
export PYTHONPATH=pipeline/shared
python pipeline/02_topic_modeling/run_6topic_discourse_cosine.py --help
```

See `manifests/script_manifest.csv` for the compact script index.

## Data Availability

Data are not included in this repository. To run the pipeline, the user must provide local corpus files, embeddings, validation decisions, review decisions, and generated intermediate files through local paths or script arguments.

Generated files should remain outside version control under ignored locations such as `data/`, `outputs/`, `pipeline/**/inputs/`, or `pipeline/**/outputs/`.
