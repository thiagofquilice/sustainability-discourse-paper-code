# Pipeline Scripts

This folder contains only the retained Chapter 4 route scripts. Use the repository root as the working directory. Each command-line script locates `pipeline/shared` automatically.

The retained folders correspond to domain assignment and BERTopic (`02_topic_modeling`), Colab/Gemma temporal synthesis (`03_topic_description_and_interpretation`), corporate-centered review (`04_classification_and_review`), minimal cross-source matching prerequisites (`05_cross_source_comparison`), and shared helpers (`shared`).

Implementation notes:
- `01_data_preparation/embed_corpus.py` creates normalized corpus embeddings required by domain assignment.
- `02_topic_modeling/run_hf_gemma_domain_validation_colab.py` uses `AutoModelForCausalLM` for binary relevance validation.
- `02_topic_modeling/materialize_adjusted_full_validation_output.py` provides the Chapter 4 T2 secondary recovery route and `generic_no_t2_recovery` for a new corpus.
- Corporate-focus review defaults to initial `0.60`, with `0.65` recorded as the primary reference threshold and not as a sole automatic selection rule.
