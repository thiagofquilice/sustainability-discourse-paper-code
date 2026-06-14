# Data Availability

No data are included in this repository.

The scripts expect locally configured external inputs such as prepared source chunks, metadata, embeddings, validation outputs, BERTopic intermediates, review workbooks or CSV decisions, and Colab/Gemma outputs. Configure those paths in `config/paper_6topic_pipeline_config.json` or pass them through script arguments.

Generated files should remain outside version control under ignored locations such as `data/`, `outputs/`, `pipeline/**/inputs/`, or `pipeline/**/outputs/`.
