# HeCBench Julia abstraction study

Technical manuscript for the PASC 2027 ACM `sigconf` format. The frozen 300- and 447-name assessments and the current 537-name scoped assessment are distinguished below. Each section has a separate file under `sections/`; detailed scientific case histories are retained in the separate technical companion.

## Current 537-name scoped assessment

The [canonical contract census](data/current_537_contract_census.json), dated 20 October 2026, records 532 accepted named contracts across 537 source benchmark names: 522 numerical, five property-equivalence and five execution-only cases; five names require unavailable native hardware. These are scoped acceptance outcomes, including documented corrections and limitations, rather than certification of every original configuration. The main 447-name statistics remain frozen; the additional 90-name extension is outside the released artifact. The active V100 roofline campaign remains incomplete and is a separate measurement assessment. Source membership proofs do not establish an observed timed invocation or complete elapsed-ROI coverage.

## Completed 447-name assessment

The completed 150-name audit adds 147 names and revisits three barriers: the union has 436 numerical contracts, two property-only contracts, five execution-only cases and four native hardware barriers, with zero pending entries. Numerical contracts include qualified corrections and known-defect reproductions. The 100/300 cohorts remain historical subsets. The October 19 revision adds independent timing blocks, verification-gate controls and normalized implementation classes without replacing frozen evidence.

## Local research evidence

`research/` is retained locally and excluded from Git tracking so GitHub and Overleaf sync only the manuscript and its supporting publication files. Existing research files remain in historical commits. Figure regeneration, evidence audits, and checkpoint restart require the local research folder; normal LaTeX compilation does not.

## Completed frozen 300 assessment

All 300 benchmark names and 900 CUDA.jl/KA/JACC implementation slots have reviewed outcomes. The study records scientific workload categories, separate implementation ontology and failure evidence for each model, and the actual supported configurations. A completed assessment does not certify every original CLI, precision or hardware mode: native reference corrections, known scientific defects, unavailable hardware/library features and unresolved original configurations remain explicit. Separately verified numerical-method studies preserve their failed primary histories. The generated census table in `sections/generated/census_progress.tex` gives the current outcome counts.

Later prospective activity windows record observed processed tokens, including repeated and cached context, coordination and disclosed helper reuse. They remain separate from the pilot's partial source estimates and unknown historical costs; overlapping phases are not additive. See the current authority at the top of the three source-repository guides and the local latest checkpoint.

## Recorded pilot experiment

- 100 benchmark names from the frozen historical 300-pass CUDA.jl population; six rejected source/hardware candidates retained in selection history.
- 200 KA/JACC implementations, 107 primary configurations, four versions per configuration: native CUDA, CUDA.jl, KA and JACC.
- Five checked trials per version; exclude trial one and average trials two–five. All samples remain available.
- 99 benchmark pairs contain translated device kernels. MRG32k3a is explicitly vendor-library interoperability, with zero translated device kernels. Backend-specific device/host extensions are separately labeled.
- Numerical audit checks 2,140 artifact entries and recomputes 428 means. Numerical equivalence applies to the documented configurations and evidence contracts, not every original CLI or precision.
- Every variant has its own source commit and attempt/cost records. All token records are **partial estimates** with heterogeneous coverage; they are not total consumed or billed tokens.

Author names, affiliations, corresponding email, publication metadata and an archival deposit remain to be supplied before submission. This repository is not a conference submission.

## Build and regenerate figures

`make` builds `build/main.pdf` with ACM `acmart` and BibTeX. The recorded build used acmart 2.20 in TeX Live 2026. The manuscript and plots contain actual measurements. Build the separate expanded report with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build technical_companion.tex`; its PDF is `build/technical_companion.pdf`. It is a technical companion, not an approved PASC supplement.

```bash
python3 -m venv .venv
.venv/bin/pip install -r research/analysis/requirements.txt
.venv/bin/python research/analysis/plot_verified_study.py
.venv/bin/python research/analysis/summarize_translation.py
.venv/bin/python research/analysis/plot_mapping_associations.py
.venv/bin/python research/analysis/plot_roofline.py
.venv/bin/python research/analysis/plot_resource_ablation.py
make
```

These figure commands use the saved portable data and do not run GPU benchmarks. The ECDF contains all primary configurations. Per-configuration sample-range plots are also exported. The resource-policy ablation retains all 17 baseline/intervention configurations, including slowdowns.

## Evidence map

| Artifact | Contents |
|---|---|
| `research/data/matched_results.json`, `.csv` | Primary configurations, four-model checks, hashes and all trial times |
| `research/data/all_matched_results.json` | Primary configurations plus valid original resource-policy ablations |
| `research/data/configuration_dispositions.json` | Explicit primary/ablation decisions, independent of preferred timing |
| `research/data/translation_accounting.json` | Raw per-variant manifests and token records |
| `research/analysis/effort_audit/` | All 200 effort slots, aliases, missing fields, shared phases and commit provenance |
| `research/data/numerical_contract_audit.json` | Numerical contracts, artifact hashes/encodings, means and scope restrictions |
| `research/data/feature_outcomes.json` | Demonstrated APIs, adaptations, backend extensions and interoperability |
| `research/data/profiling/` | Separate raw counter CSVs and derived roofline/resource measurements |
| `research/observations/` | Benchmark-specific RQ observations and confounds |
| `research/checkpoints/LATEST.json` | Combined source/paper restart checkpoint |

Full binary outputs, local SQLite results and raw compiler logs remain under the source checkout's `.porting-state/`; portable snapshots retain their hashes and relevant summaries. Numerical/effort audit scripts require this local state. They do not manufacture missing evidence or collect new GPU series. Profiled device durations are separate from required host-inclusive trial means.

## Source checkout and fresh replication

The separate HeCBench source repository contains nested implementations under `src/<bench>-julia/<bench>-julia-{cuda.jl,KA,JACC}/`, pinned Julia environments, benchmark configurations, per-variant `verification.json`, and the ontology study layer under `ontology/studies/julia_abstractions/`. The latter archives inferred base YAML hashes/snapshots and reviewed configuration traits without claiming that inferred labels are measured bottlenecks.

Use the exact source revision and a primary configuration listed in the result snapshot. For a relocated checkout, `tools/prepare_portable_study_rerun.py --config CONFIG --output REPLICATION.json --gpu DEVICE` relocates source paths, removes archived sample reuse, and assigns a fresh replication ID. Then `python3 tools/verify_portable_julia.py --config REPLICATION.json` collects a separate four-model series. Install the pinned Julia environments, select the correct NVIDIA device, and retain declared flags/scopes. Do not replace the frozen paper measurements with preferred rerun results.

## Restart and interpretation

Start with the source repository's `AGENT_INSTRUCTIONS_CODEX_Julia.md`, then its linked `JULIA_CORRECTNESS_CHECK.md` and `paper_writing.md`, followed by `research/checkpoints/LATEST.json`. Preserve separate source and paper Git histories.

The cohort is feasibility-selected, not random. Native/Julia algorithms, layouts, allocation policies, arithmetic and timing scopes sometimes differ and are disclosed. CUDA-only results do not establish cross-vendor portability. Four retained samples do not establish a universal performance ranking, and partial token proxies cannot rank total translation cost.
