# Minimal Route Summary

The retained code covers the Chapter 4 route from a prepared corpus through temporal synthesis. It is now aligned to the audited historical behavior where the server evidence was conclusive.

1. Prepared corpus embedding. `pipeline/01_data_preparation/embed_corpus.py` is a compatible utility; the exact historical embedding creation remains inferred from metadata and downstream scripts.
2. Six-domain catalog and cosine assignment. Historical positives require `best_score >= 0.65`; `score_gap`/`score_margin=0.02` is diagnostic unless explicitly enabled for sensitivity analysis.
3. Binary Gemma validation using the sanitized historical `AutoModelForCausalLM` runner.
4. Historical T2 secondary recovery materialization. The generic no-T2 route is available only as an explicit mode for new data.
5. Source-domain BERTopic microtopic modeling across 18 source-domain subgroups with the audited parameters preserved.
6. Human microtopic review and consolidation. The generic workbook builder is an auxiliary template when original review decisions are unavailable.
7. Cross-source semantic comparison.
8. Corporate-centered topic review and selection. Initial inclusion threshold is `0.60`; `0.65` is the main reference threshold and manual review/overrides are part of the route.
9. Colab/Gemma annual and temporal synthesis. The audited intermediate input had `241` topics and `4344` topic-year evidence rows.
10. Post-synthesis review and final group exclusion. Audited final manifests record `182` retained groups: `18` corporate, `12` media, and `152` academic, with `114047` unique documents.

Later figure/table rendering, temporal diagnostics, interface materials, and complementary analyses were removed from this minimal code supplement. A final `153` academic / `165` noncorporate count was not confirmed by the audited manifests and is therefore not represented as a verified route checkpoint here.
