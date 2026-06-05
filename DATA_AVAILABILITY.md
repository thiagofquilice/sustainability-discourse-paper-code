# Data Availability

This repository does not bundle the source corpora or generated data products.
It contains the paper-facing code, prompts, example configuration, and workflow
notes.

## Not Bundled

The following file types and folders are not included:

- raw corporate filings, academic records, or media articles;
- full source text, representative-document exports, or evidence workbooks;
- document embeddings, profile embeddings, NumPy arrays, and vector files;
- BERTopic model folders and serialized model artifacts;
- generated pipeline outputs, paper tables, and rendered figures;
- Colab input/output bundles and restored download staging folders.

## Expected External Inputs

Scripts refer to external inputs such as:

- prepared source corpora and metadata tables;
- descriptor catalog and embedding assignment outputs;
- LLM validation outputs;
- BERTopic document-topic and topic-info tables;
- corporate review decisions and override tables;
- cross-source matching outputs;
- final annual prevalence and temporal relation inputs.

`config/paper_6topic_pipeline_config.example.json` shows the expected style of
path configuration.

## Data Placement For Local Review

If compatible files are available locally, place them outside the repository or
under ignored folders such as:

```text
data/
outputs/
pipeline/**/inputs/
pipeline/**/outputs/
```

These folders are ignored by Git to reduce the risk of accidentally publishing
large files or source text.

