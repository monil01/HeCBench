# Final census and manuscript consistency review

Read-only review of current compact census, extension records, abstract, main extension/conclusion/artifact text and companion extension prose. No GPU, source/state/paper edits or Git operations. Exact reviewed file hashes and arithmetic recounts are in `review.json`. This review does not independently rerun the global primary-artifact audit or rederive the frozen377-name performance analysis.

## Counts reconcile

| Scope | Numerical | Property | Execution only | Hardware barrier | Total |
|---|---:|---:|---:|---:|---:|
| Frozen study | 436 | 2 | 5 | 4 | 447 |
| Additional names | 86 | 3 | 0 | 1 | 90 |
| Full catalogue | 522 | 5 | 5 | 5 | 537 |

The current census contains exactly532 rows and532 unique accepted names. Recounting every row reproduces the outcome and qualification summaries. The five unaccepted names are exactly blas-fp4gemm, blas-fp8gemm, blas-mxfp6gemm, blas-mxfp8gemm and twell. They have terminal hardware assessments, not unresolved target implementation progress.

The separate word2vec workflow is one accepted additional **configuration**, not an extra name or a replacement for the numerical component in the primary name census. Its property-only status and distinct trained models are explicit. Current abstract and main/companion extension prose contain no blanket claim of537 numerically equal full applications and no stale claim that this bounded workflow is still wholly unverified. Text8/production quality and full raw embedding identity remain correctly uncertified.

The final saved audit's10,640 trial rows equal532×4×5. Its10,540 primary artifact paths equal527×4×5, excluding the five execution-only cases. Its zero hard-issue count and stated full-byte/recorded-gate scope are consistent with the artifact appendix; auxiliary word2vec scientific properties have their own separate audit. The frozen release and later extension are explicitly distinguished.

## Two minor wording suggestions

1. `sections/09_conclusion.tex`: “retaining open and hardware-limited assessments” can suggest names remain unassessed. Replace with “retaining hardware-limited assessments and explicit scope gaps.”
2. `data/prospective_extension.json`, `count_scope`: “not universal scientific validity or completion of537” is ambiguous next to the manuscript's completed scoped assessments. Replace the latter clause with “not full original-CLI/scientific validation of537.”

These are descriptive refinements, not count, acceptance or scientific-contract failures. No material contradiction found.

## Accounting

Separate shared review BEGIN `d6b0200aa144ce5d088ad614`, phase `final_census_paper_consistency_review`, benchmark `study537`. Closed END and combined observed counters are recorded in `usage_receipt.json`. Context/cache, coordination and reporting are included; no retrospective per-benchmark or per-model translation-cost allocation is made.
