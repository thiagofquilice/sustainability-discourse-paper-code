# Audit Corrections Manifest - 2026-08-15

This manifest records the read-only server evidence used to correct the PR branch. It deliberately omits corpus text, credentials, tokens, and private keys.

## Proven By File Or Manifest

### 411658 versus 410658 rows

- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/cosine/assignment_diagnostics.json`, field `positive_rows=411658`, `threshold=0.65`, `variant=six_topic_elements_max_without_margin`, SHA256 `2342f16844b6597b87ed817092e06a23b16d47a40d53865f4ce65fc31a363229`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/scripts/build_6topic_pilot_and_colab_package.py`, lines 199 and 226-227 remove the pilot `chunk_id` values from the full input and record `excluded_pilot_rows`, respectively.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/pilot_1000/input/six_topic_pilot_1000_manifest.json`, fields `rows=1000`, `source_quotas={academic:334, corporate:333, media:333}`, `seed=42`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/full_run/input/six_topic_best_only_remaining_manifest.json`, fields `rows=410658`, `excluded_pilot_rows=1000`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/full_run/best_only/run_manifest.json`, fields `total_input_rows=410658`, `rows_in_output=410658`, `error_rows=0`, `model_name=google/gemma-4-E4B-it`, `transformers_version=5.5.3`, SHA256 `b73dccc58b3eea4ed96dfed5b4da339639c65fb987bc59c9898a4e4582446d4d`.
- Identification of the 1000 rows: the exact rows are the records in `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/pilot_1000/input/six_topic_pilot_1000.parquet` and `.csv`. Sorted `chunk_id||assigned_label` SHA256 is `36f3b9accaf30a2dc263419118060370437fc4b8dedb74e464dea1f5cbc2d96b`; sorted `chunk_id` SHA256 is `d3a1a1921a41634924b6e1f3d97d33aee4d6cc35ea147d17b0be8ed3a9012d75`.

Conclusion: the 1000 rows were not lost. They were split into the pilot package before the full Gemma input was built, which explains 411658 positive assignments versus 410658 full-run validation/materialization rows.

### T2 secondary recovery nine-row difference

- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/t2_secondary_recovery_cosine_round/summary.json`, fields `old_t2_rows=56817`, `changed_to_other=5333`, `changed_to_outside=12602`, `remain_t2=38882`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/full_run/adjusted_with_t2_secondary_recovery/adjustment_manifest.json`, fields `full_t2_to_other=5324`, `expected_changed_rows=5333`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/full_run/adjusted_with_t2_secondary_recovery/adjusted_full_manifest.json`, fields `changed_gemma_rows_used=5333`, `adjusted_rows=398109`, `adjusted_yes_rows=353971`, SHA256 `d9ff994e663c21f4c715c53e7ffc457ccb07fa557f28313e401d4d2606ba169d`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/scripts/materialize_adjusted_full_validation_output.py`, lines 107-145 concatenate all `changed_gemma` rows that share output columns, while lines 123-127 compute full-run T2 transitions only from full input rows; SHA256 `eb620863ee8801f5addcf171f000c44971f5bda74f58c7b87ba2c3e6c47a0ede`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/scripts/run_6topic_micro_unsupervised.py`, line 23 default input is `/outputs/full_run/adjusted_with_t2_secondary_recovery/adjusted_full_yes.csv`; SHA256 `791af7b1ed34939bc3a282c9828987d4f11129d867258dbae5072e608be66390`.

The nine pilot old-T2-to-other rows are identified by these `chunk_id` values, with no corpus text included:

```text
paper::54e3409b3b4114851ddda87c2f8a0a7da41a61f9::chunk_0001
paper::6e2e61a2f18201c20d300d62bb6afc91aaec478b::chunk_0001
paper::cbce48e8a99f1524f297f54d373b62f890e75f94::chunk_0001
paper::092e20e0bf298268043630355ea7258396c77bbf::chunk_0001
paper::6c8a10dd51c18a0266cebfd87b3acdd83cb913a9::chunk_0001
10k::NFTN-0001165527-17-000188-1A::chunk_0008
10k::AIVN-0000005656-18-000022-1A::chunk_0012
guardian::sustainable-business/singapore-sustainable-data-centres::row_455797::chunk_0001
guardian::environment/2022/mar/12/no-10-must-not-cave-in-to-tory-climate-sceptics-on-fracking-says-ed-miliband::row_742092::chunk_0001
```

Sorted nine-row `chunk_id||new_label` SHA256: `7061129a3671094c485392fcf96628d474291bcea27aea53390587c0e004ce08`.

Conclusion: the nine-row difference was caused by pilot rows present in the T2 recovery universe but absent from the 410658-row full Gemma input. It was not caused by duplicated identifiers, error rows, or failed resumes. The 18 BERTopic models used `adjusted_full_yes.csv`, which includes the materialized T2 secondary recovery route.

### 241 topics, 4344 topic-year rows, and final repertoire

- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/corporate_focus_stage12_input_with_overrides/selection_manifest.json`, fields `selected_micro_topic_count=241`, `annual_evidence_row_count=4344`, `included_corporate_group_count=27`, `included_noncorporate_group_count=214`, SHA256 `5a3b7c1ab6ba0413f5950c07180a888866178975fb70c45aa36a9816bb144e0c`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/corporate_focus_review_with_overrides/included_noncorporate_groups.csv`, rows `214`, split `academic=197`, `media=17`, with reasons `direct_pair_with_corporate_above_threshold=170` and `manual_override_from_excluded_review=44`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/corporate_focus_review_with_overrides/corporate_focus_commented_decision_summary.json`, fields `total_master_review_rows=214`, `delete_rows=44`, `not_related_rows=18`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/corporate_external_group_exclusion_review/group_exclusion_materialization_manifest.json`, fields `selected_groups_pre_filter=197`, `selected_groups_retained=182`, source counts `academic retained_groups=152 retained_unique_documents=97762`, `media retained_groups=12 retained_unique_documents=11012`, `corporate retained_groups=18 retained_unique_documents=5273`, SHA256 `574bcb6f24d564a0f31d7ea606d0148a40b123c60ceb6f74af8ff22f1587b4ec`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/paper_tables/corporate_focus_filter_stage_counts/filter_stage_source_counts.csv`, final row records `academic=97762`, `media=11012`, `corporate=5273`, `total=114047` unique documents.

Conclusion: the audited final group-exclusion manifests prove `18` corporate, `12` media, and `152` academic retained groups, with `114047` unique documents. A final `153` academic / `165` noncorporate count was not confirmed by these manifests and would conflict with the source-count manifest.

### Claim of 232 retained noncorporate topics

- Evidence: manifests searched for `232` did not contain a final noncorporate-retained count of 232.
- Evidence: a small file with exactly 232 rows exists at `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/bertopic_micro_unsupervised_multiaspect/academic_T3/hierarchical_topics.csv`, but that is a BERTopic hierarchical-topic output for one subgroup, not a retained noncorporate repertoire count.
- Evidence: `241 - 9 = 232` can be derived by subtracting the nine excluded corporate anchors from the synthesis input, but that derived value still includes corporate and all 214 noncorporate synthesis topics, so it is not a final noncorporate count.

Conclusion: `232 retained noncorporate microtopics` is not supported as a final or manifest-backed count. The confirmed noncorporate checkpoints are `214` sent to synthesis/review, `170` retained after comment review before final group exclusion, and `164` retained after the final group-exclusion manifest (`152` academic + `12` media).

### ODS/SDG crosswalks

- Evidence: real catalog `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/catalog/six_topic_discourse_catalog.csv`, SHA256 `2ceb81fe7cf0ab9eb61f2a6fc29c8e6c38a8fee21ef2b83c6526cc2d725681af`.
- Evidence: supplement catalog `catalog/six_topic_discourse_catalog.csv` has the same SHA256 `2ceb81fe7cf0ab9eb61f2a6fc29c8e6c38a8fee21ef2b83c6526cc2d725681af`.
- Evidence: Appendix I workbook `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/paper_tables/appendix_method_tables/paper_appendix_method_tables.xlsx`, sheet `Macro topics`, has the same six `sdg_crosswalk` values; workbook SHA256 `7f9f6fb9816d648836627737604d4841ee2a2d001549841f1266189fe1bf854c`.
- Evidence: `/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/scripts/run_6topic_discourse_cosine.py`, lines 56-68 build subanchors with `subanchor_text` as the descriptor expression, line 114 embeds `subanchor_text` with `normalize_embeddings=True`, lines 156 and 185 copy `sdg_crosswalk` to outputs as metadata.

Conclusion: SDG/ODS associations did not participate in six-domain embedding generation. They functioned as metadata in outputs and appendix tables.

## Inferred

- The exact historical corpus-embedding creation command remains inferred. The audited metadata and classification script prove `BAAI/bge-large-en-v1.5`, 1024 dimensions, and normalized downstream use; `pipeline/01_data_preparation/embed_corpus.py` is therefore documented as a compatible utility, not as a proven historical script.
- Current local environment package versions are not proof of the historical BERTopic runtime. The BERTopic manifests did not store versions for BERTopic, sentence-transformers, torch, spaCy, UMAP, or HDBSCAN.

## Not Possible To Determine From Available Manifests

- Exact historical versions of BERTopic, sentence-transformers, torch, spaCy, UMAP, and HDBSCAN. Gemma manifests do record Transformers versions: `5.5.3` for six-domain validation and `5.5.4` for temporal synthesis.
- A manifest-backed final count of `153` academic or `165` noncorporate retained topics.
- A manifest-backed final count of `232` retained noncorporate microtopics.

## Corrections Applied In This PR Branch

- Restored historical six-domain positive mask to `best_score >= similarity_threshold`; `score_gap` remains diagnostic and the margin sensitivity option defaults to false.
- Added explicit config fields documenting that the score margin was not applied to the historical positive mask.
- Replaced the reconstructed six-domain Gemma validation script with the sanitized historical `AutoModelForCausalLM` runner.
- Restored `AutoModelForCausalLM` in temporal Gemma runners and set the bundle Stage 2 `max_new_tokens` to the observed output-manifest value `650`.
- Restored `0.60` as the initial corporate review threshold and documented `0.65` as primary/reference threshold with manual review and overrides.
- Preserved T2 secondary recovery as the default historical materialization route and kept a generic no-T2 route as an explicit option.
- Documented `241` topics and `4344` topic-year rows as temporal synthesis inputs and the later reduction to the manifest-proven final repertoire.
- Kept BERTopic parameters consistent with the audited script/manifests.
- Marked `embed_corpus.py` and the generic merge-review workbook builder as compatible/auxiliary rather than exact historical evidence.

## Supplement File Hashes After Correction

| File | SHA256 |
|---|---|
| `config/paper_6topic_pipeline_config.example.json` | `b8259edbbc80afd4fdad516a6f7e27a3abff4802c931d9211505593efbcc6d9f` |
| `pipeline/02_topic_modeling/run_6topic_discourse_cosine.py` | `ac5cf8d9a3dfc92da031e275d3bd7550d743ec734162182d9718f620e2ce1fce` |
| `pipeline/02_topic_modeling/run_hf_gemma_domain_validation_colab.py` | `f0f8241b09ce96f9a46f633ce7a5a54c1da52c74b3d453145c0f8bfc2e3566cb` |
| `pipeline/02_topic_modeling/materialize_adjusted_full_validation_output.py` | `4130df6865f257194172236d667299b082b36e2aa6c0644f6f4ecf0c21d62492` |
| `pipeline/03_topic_description_and_interpretation/run_hf_gemma_micro_topic_year_summaries_colab.py` | `0f77dcc7d16cf165e4104eb1a8ac42071f32c7498f892effa9ae2ccc4d1b3176` |
| `pipeline/03_topic_description_and_interpretation/run_hf_gemma_micro_topic_evolution_synthesis_colab.py` | `0e05ddf166179644518208cc611bf6c379f9055d596f1e5534adf22345e855d5` |
| `pipeline/03_topic_description_and_interpretation/build_corporate_focus_drive_colab_bundle.py` | `416060d60ad5a6b1095c805e94469a158c3dd86a70ee8ee5282ef67d2b72b081` |
| `pipeline/04_classification_and_review/build_corporate_focus_review.py` | `11002ead67750f85c0c52c2e18f1028a0bafcf1a28938dee54bc843be24bba19` |
| `manifests/route_summary.md` | `58a9e6f2ea7837c271832405b7cfc54c57237d9a8e84eff6f0d1c77a1b446d49` |
| `manifests/script_manifest.csv` | `7a57feed8c3f2d83bee4efecc0e17d5db4cb48f55bb3e3b3bfaebef0ea87e711` |
