# Reproducibility Notes

This is a code-only supplement. To run it, provide local external data and generated intermediate files that match the schemas expected by each script.

## Setup

```bash
cp config/paper_6topic_pipeline_config.example.json config/paper_6topic_pipeline_config.json
# edit config/paper_6topic_pipeline_config.json with local paths
export PYTHONPATH=pipeline/shared
```

## Command Skeleton

```bash
python pipeline/02_topic_modeling/run_6topic_discourse_cosine.py --config config/paper_6topic_pipeline_config.json
python pipeline/02_topic_modeling/materialize_adjusted_full_validation_output.py
python pipeline/02_topic_modeling/run_6topic_micro_unsupervised.py
python pipeline/02_topic_modeling/materialize_microtopic_group_review_mapping.py
python pipeline/02_topic_modeling/build_merged_microtopic_root.py
python pipeline/05_cross_source_comparison/build_cross_source_microtopic_profiles.py
python pipeline/05_cross_source_comparison/embed_cross_source_microtopic_profiles.py
python pipeline/05_cross_source_comparison/match_cross_source_microtopics.py
python pipeline/04_classification_and_review/build_corporate_focus_review.py
python pipeline/03_topic_description_and_interpretation/build_corporate_focus_stage12_inputs.py
```

The Colab/Gemma scripts in `pipeline/03_topic_description_and_interpretation/` are intended for GPU runtimes or local environments with the required model access.
