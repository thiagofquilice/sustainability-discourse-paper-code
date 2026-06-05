# Reproducibility Notes

This repository provides a stage-organized view of the code used in the final
paper route. The route is summarized as:

```text
Unified six-topic pipeline
+ corporate-focus branch with manual overrides
+ post-comment paper results package
```

## How To Navigate The Code

Use `manifests/script_manifest.csv` as the main index. It links each script to a
workflow stage, expected inputs, expected outputs, and its role in the paper
route.

The pipeline stages are:

1. Corpus and sample construction.
2. Six-topic assignment and BERTopic microtopic modeling.
3. LLM-assisted annual summaries and temporal evolution narratives.
4. Corporate-centered review, override, and category decisions.
5. Cross-source microtopic matching and pair diagnostics.
6. Longitudinal prevalence and temporal relation analyses.
7. Complementary corporate exclusions and dual tables.
8. Final paper tables, figures, and appendix materials.

## Running Scripts

Most scripts expect external inputs that match the original project workspace
schemas. To adapt them:

1. Copy `config/paper_6topic_pipeline_config.example.json`.
2. Replace placeholder paths with local paths.
3. Run scripts from the repository root or add `pipeline/shared/` to
   `PYTHONPATH` when inspecting stage scripts that import shared helpers.

Example:

```bash
PYTHONPATH=pipeline/shared python pipeline/06_temporal_analysis/build_corporate_focus_relative_longitudinal_series.py --help
```

LLM and Hugging Face scripts may require a GPU runtime and model access. Their
prompt templates are included under `docs/method/prompts/`.

## Included Route Materials

The repository includes code and documentation for the final route only. Earlier
pilots, smoke tests, restored download staging folders, broad exploratory
branches, and abandoned analyses are not included in the stage folders.

## Public-Facing Limits

The repository is intentionally code-centered. It does not bundle raw text,
large embeddings, generated model folders, or paper result outputs. The expected
input locations and excluded file classes are described in
`DATA_AVAILABILITY.md`.

