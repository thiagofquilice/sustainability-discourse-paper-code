# Minimal Route Summary

The retained code covers the Chapter 4 route from a prepared corpus through temporal synthesis.

1. Prepared corpus embedding with `BAAI/bge-large-en-v1.5` and normalized float32 output.
2. Six-domain catalog and cosine assignment. Positive assignments require `best_score >= 0.65`; `score_gap`/`score_margin=0.02` is diagnostic unless explicitly enabled for sensitivity analysis.
3. Binary Gemma validation using `AutoModelForCausalLM`.
4. Study-specific T2 secondary recovery materialization. A generic no-T2 mode is available for new data.
5. Source-domain BERTopic microtopic modeling across the 18 possible source-domain subgroups.
6. Human microtopic review and consolidation. The generic workbook builder is an auxiliary template when original review decisions are unavailable.
7. Cross-source semantic comparison.
8. Corporate-centered topic review and selection. Initial inclusion threshold is `0.60`; `0.65` is the main reference threshold and manual review/overrides are part of the route.
9. Colab/Gemma annual and temporal synthesis.
10. Post-synthesis review and final group exclusion.

Empirical inputs, generated outputs, filled review workbooks, figures, tables, interface materials, and complementary analyses are not included in this code supplement.
