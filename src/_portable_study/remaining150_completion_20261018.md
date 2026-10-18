# Remaining 150: completed assessment, 18 October 2026

The frozen 150-name worklist has no pending entries:

| Outcome | Names |
|---|---:|
| Numerical agreement under recorded output contracts | 139 |
| Property checks only (`atomicSystemWide`, `rsc`) | 2 |
| Execution checks only (`assert`, `dispatch`, `kernelLaunch`, `mallocFree`, `pointerchase`) | 5 |
| Native hardware barriers | 4 |

The numerical group includes 27 corrected-native outcomes, four known-native-defect reproductions, and one revised-acceptance outcome. These qualifications remain explicit; this is not a claim that all 150 are numerically correct applications or that all physical raw outputs are identical.

Across the combined 447-name population, 436 have numerical contracts, two have property contracts, five have execution contracts, and four have native hardware barriers. The checkpoint's 438 accepted numerical/property names includes the two property-only names.

## Evidence and timing

Each accepted timing series uses five checked trials, discards the first timing, and averages trials two through five. KA and JACC verified implementations have separate source commits. The final numerical audit checks 501 configurations and 10,020 saved artifact hashes with zero hard issues. The effort audit finds 443 working names per portable model matching committed source, with zero provenance flags. Missing historical token/attempt evidence remains unknown. Shared coordination and nested translation usage windows overlap and must not be added together.

The four barriers are `blas-fp4gemm`, `blas-fp8gemm`, `blas-mxfp6gemm`, and `blas-mxfp8gemm`. Actual native builds, guards, descriptor queries, or library diagnostics establish the hardware limitations. Revisit with a suitable GPU/software environment; these are not demonstrated KA/JACC feature failures and have no successful performance measurements.

## Paper and restart

The updated PASC manuscript is `2026-PASC-Hecbench-Julia-paper/build/main.pdf` (10 body pages plus references); the separate technical companion has 36 pages. Publication sources and compact figures are pushed to the paper repository. Research data, full captures, and large generated CPU fixtures stay local and outside GitHub/Overleaf. Author/publication metadata and archival release remain pending.

Restart through `AGENT_INSTRUCTIONS_CODEX_Julia.md`, `JULIA_CORRECTNESS_CHECK.md`, and `paper_writing.md`, then inspect `2026-PASC-Hecbench-Julia-paper/research/checkpoints/LATEST.json` and `.porting-state/portable_julia/active_work_447.json`. Preserve completed evidence, frozen populations, and historical raw events. Original ASSERT negative-child text was lost; separately retained new corroboration logs are not a recovery of the original executions. Source-bound recipes and hash proofs document excluded generated fixtures.
