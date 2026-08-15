# Sustainability Discourse Paper Code

This repository is the code supplement for Chapter 4 of the thesis. It covers the computational route from a prepared three source corpus to six environmental domain assignments, validated source and domain corpora, source specific BERTopic microtopics, reviewed topic consolidation, corporate centered cross source comparison, and Gemma assisted temporal synthesis.

Empirical data and generated research outputs are excluded. A reader can run the code with another corpus that follows the documented schema.

## Methodological scope

The retained implementation follows these Chapter 4 parameters.

| Component | Published specification |
|---|---|
| Embeddings | `BAAI/bge-large-en-v1.5` with 1024 dimensions |
| Domain assignment | Best cosine similarity of at least `0.65` and a margin of at least `0.02` over the second domain |
| Relevance validation | `google/gemma-4-E4B-it`, binary `yes` or `no`, `max_new_tokens=1`, deterministic decoding |
| BERTopic models | One model for each of the 18 source and domain subgroups |
| Vectorization | English stop words, `min_df=1`, `max_df=0.95` |
| Topic representation | KeyBERTInspired, part of speech filtering, and Maximal Marginal Relevance |
| Cross source selection | At least one direct relation to a corporate microtopic with cosine similarity of at least `0.65`, followed by human review |

The source values expected by the scripts are `academic`, `media`, and `corporate`. The environmental domain codes are `T1` through `T6`.

## Installation

Python 3.11 is recommended. Run the following commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-full.txt
python -m spacy download en_core_web_sm
```

The command-line scripts locate the shared helper directory automatically when they are run from the repository root.

Copy the example configuration and edit the local paths when needed.

```bash
cp config/paper_6topic_pipeline_config.example.json config/paper_6topic_pipeline_config.json
```

The Gemma stages require access to the model on Hugging Face and a compatible GPU runtime. The current Gemma 4 implementation uses the Transformers multimodal processor and model classes. CPU only readers can run the preparation, assignment, review, and most downstream materialization steps with a small corpus, although embedding and BERTopic execution will be slower.

## Prepared corpus schema

The prepared corpus may be a Parquet or CSV file for embedding creation. Domain assignment reads Parquet. It requires these columns.

| Column | Meaning |
|---|---|
| `doc_id` | Stable document identifier |
| `chunk_id` | Stable and unique text unit identifier |
| `text` | Text used for embedding, classification, and topic modeling |
| `source` | One of `academic`, `media`, or `corporate` |
| `year` | Integer publication or filing year |
| `industry` | Industry label or an empty string when it does not apply |
| `source_doc_id` | Source level document identifier used for document counts |

Corpus row order must remain stable between embedding creation and domain assignment. The embedding metadata record the model, number of rows, and dimensions. Domain assignment stops when the configured model and metadata model differ.

## Pipeline route

| Step | Purpose | Main script | Human or external dependency |
|---|---|---|---|
| 1 | Create normalized corpus embeddings | `pipeline/01_data_preparation/embed_corpus.py` | Prepared corpus |
| 2 | Assign chunks to the six domains | `pipeline/02_topic_modeling/run_6topic_discourse_cosine.py` | Local config and embeddings |
| 3 | Validate provisional assignments | `pipeline/02_topic_modeling/run_hf_gemma_domain_validation_colab.py` | Hugging Face model access and a GPU |
| 4 | Materialize the validated corpus | `pipeline/02_topic_modeling/materialize_adjusted_full_validation_output.py` | Validation output and optional study specific T2 adjustment inputs |
| 5 | Fit the 18 source and domain BERTopic models | `pipeline/02_topic_modeling/run_6topic_micro_unsupervised.py` | Validated corpus, embeddings, and `en_core_web_sm` |
| 6 | Prepare and materialize topic merge review | `pipeline/02_topic_modeling/build_microtopic_merge_review_workbook.py` and `pipeline/02_topic_modeling/materialize_microtopic_group_review_mapping.py` | Human review of the workbook |
| 7 | Build the reviewed merged microtopic root | `pipeline/02_topic_modeling/build_merged_microtopic_root.py` | Reviewed mapping |
| 8 | Build, embed, and compare cross source profiles | Scripts in `pipeline/05_cross_source_comparison` | Sentence transformer model |
| 9 | Construct the initial corporate centered repertoire | `pipeline/04_classification_and_review/build_corporate_focus_review.py` | Human review of corporate and external topic decisions |
| 10 | Prepare and run temporal synthesis | Scripts in `pipeline/03_topic_description_and_interpretation` | Hugging Face model access and a GPU |
| 11 | Audit the repertoire with temporal narratives | `pipeline/04_classification_and_review/build_corporate_focus_hybrid_review_workbook.py` | Completed Stage 2 narratives and human review |

The merge review workbook begins with every non outlier microtopic listed as a singleton. This is an explicit pending review state. The reviewer moves candidate groups into the two group sheets and records `accept`, `reject`, or `split` decisions before materialization.

The initial corporate review includes all corporate groups and selects academic or media groups when any direct corporate relation reaches `0.65`. After temporal synthesis, the hybrid workbook brings the Stage 2 narratives back into the review so the researcher can compare corporate and noncorporate interpretations.

## Quick structural check

The following command verifies that every Python file compiles without downloading research models.

```bash
python -m compileall -q pipeline
```

The full command sequence and the boundaries between automated and reviewed stages are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The compact script index is in [manifests/script_manifest.csv](manifests/script_manifest.csv).

## Data availability

The original corpus, embeddings, validation outputs, filled review workbooks, fitted BERTopic models, Gemma outputs, figures, and final tables are not included. Generated files should remain under ignored local directories such as `data`, `outputs`, and stage specific input or output folders.
