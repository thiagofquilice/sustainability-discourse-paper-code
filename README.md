# Sustainability Discourse Paper Code

Code supplement for the paper's corporate-centered analysis of sustainability
discourse across corporate, academic, and media sources.

This repository collects the paper-facing scripts, prompt templates, example
configuration files, and concise workflow notes used to organize the final
analysis route. It is designed for reviewers and readers who want to inspect how
the paper pipeline was assembled.

## Repository Structure

```text
catalog/       Six macro-topic descriptor catalog.
config/        Example configuration with placeholder paths.
docs/          Method notes and LLM prompt templates.
manifests/     Script and route manifests for navigating the code.
pipeline/      Stage-organized scripts from the final paper route.
```

The `pipeline/` folder follows the order of the paper workflow:

1. `01_corpus_and_sample`
2. `02_topic_modeling`
3. `03_topic_description_and_interpretation`
4. `04_classification_and_review`
5. `05_cross_source_comparison`
6. `06_temporal_analysis`
7. `07_corporate_complementary_analysis`
8. `08_paper_tables_and_figures`

Shared helper modules are placed in `pipeline/shared/`.

## Suggested Reading Order

1. Start with `docs/method/01_method_stages.md`.
2. Review the six-topic catalog in `catalog/`.
3. Read `REPRODUCIBILITY.md` and `DATA_AVAILABILITY.md`.
4. Use `manifests/script_manifest.csv` to navigate the stage scripts.
5. Inspect LLM prompts in `docs/method/prompts/`.

## Dependencies

The full upstream workflow used common scientific Python libraries together
with heavier optional dependencies for embedding models, BERTopic, and LLM/GPU
steps. The broad dependency file is:

```bash
pip install -r requirements-full.txt
```

Individual scripts may require external files, credentials, or local paths not
included in this repository. Use `config/paper_6topic_pipeline_config.example.json`
as a template for path configuration.

## Notes

- Raw corpora, embeddings, model folders, notebooks, workbooks, and generated
  outputs are not bundled here.
- Scripts are organized by the final paper route, not by every experiment that
  was attempted during the project.
- The Streamlit reader companion from the broader project is not included as
  part of this code supplement.

