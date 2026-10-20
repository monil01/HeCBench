#!/usr/bin/env python3
"""Publish final assessment guidance only after a complete, integrity-checked census."""
from pathlib import Path
import datetime,json
R=Path(__file__).resolve().parents[3]
census=json.loads((R/'src/_portable_study/completion537_20261020/current537_census.json').read_text())
audit=json.loads((R/'2026-PASC-Hecbench-Julia-paper/research/data/numerical_contract_audit_537.json').read_text())
assert census['native_names']==537 and census['accepted_named_contract_names']==532
assert census['exclusive_outcome_counts']=={'named_numerical':522,'property_equivalence':5,'execution_only':5}
assert sorted(census['no_aggregate_eligible_accepted_contract_names'])==['blas-fp4gemm','blas-fp8gemm','blas-mxfp6gemm','blas-mxfp8gemm','twell']
assert audit['summary']['hard_issue_count']==0 and audit['summary']['selected_benchmark_names']==532
assert audit['summary']['audited_benchmarks']==527
block='''## Current full-suite assessment — 20 October 2026

All **537 native CUDA names have completed scoped assessments**. **532 names have accepted all-four contracts**: 522 named numerical, five property-equivalent and five execution-only. The other five names (`blas-fp4gemm`, `blas-fp8gemm`, `blas-mxfp6gemm`, `blas-mxfp8gemm`, `twell`) have fifteen explicit hardware/native-backend-limited model assessments; they are unverified, have no accepted performance samples, and do not demonstrate KA/JACC language limitations. Resume hardware cases only when the unchanged native contract can execute on authorized supporting hardware/library versions. The final hardware review separately identifies source architecture guards, unsuccessful exact-descriptor library queries and actual incompatible device images.

Resume translation and manuscript through `2026-PASC-Hecbench-Julia-paper/research/checkpoints/LATEST.json`; inspect `src/_portable_study/completion537_20261020/current537_census.json` and the hash-bound final primary-artifact integrity report before making coverage claims. The joint checkpoint records source/paper commits, all configuration evidence, open meters, storage state and next actions. These documents remain linked: `AGENT_INSTRUCTIONS_CODEX_Julia.md`, `JULIA_CORRECTNESS_CHECK.md`, and `paper_writing.md`. Earlier active/cohort notes below are historical; the 300/447 histories and public artifact v1.0.0 remain frozen.

A named numerical contract can be a limited component, corrected reference, revised acceptance, known-native-defect reproduction or an integration/multi-device configuration. It does not certify a complete scientific application. Word2vec retains its numerical CBOW component and additionally has a complete bounded asynchronous property workflow under `src/word2vec-julia/async_workflow` and `ontology/studies/julia_abstractions_537/workflow_studies/word2vec`; all twenty trained models differ, and this additional configuration adds no benchmark name. The fresh guarded t-SNE I1 contract passes; the strict I3 rejection and unverified original I1000 trajectory remain explicit. MMCSF's original Float32 rejection remains separate from its accepted mixed-precision secondary. Future deeper trajectories, other datasets/sizes/precisions and independent scientific quality checks require new declared contracts and fresh verification.

Every accepted newly measured version retains all five correctness checks, dual native-CUDA/CUDA.jl comparisons and independent checks, complete declared actual inputs/outputs/guards, source/environment bindings, and five timings averaged only over runs 2–5. Preserve failures, program/device-JIT/mathematical-candidate units, and actual prospective token activity windows. Unknown historical actual tokens remain unknown; forecasts and source-length proxies are separate. Cached input and reasoning counters are subsets. Shared preparation, review, annotation, coordination and paper windows can overlap target windows and must not be summed as exclusive translation costs. Root alone commits source; each verified implementation remains separately committed.

New large evidence is partly stored in owned `/tmp` trees with canonical aliases and exact path/hash provenance. **Preserve those captures before reboot or temporary-storage cleanup.** Missing raw captures cannot be certified by their hashes alone. Keep large research and GPU captures outside the Overleaf paper Git repository; compact data and PDF recipes are versioned. The later extension and its local raw evidence are not implicitly included in release v1.0.0. Never invoke sudo/sudoedit or any privilege-escalation workaround; no OS update is required.

'''
for name in ['AGENT_INSTRUCTIONS_CODEX_Julia.md','JULIA_CORRECTNESS_CHECK.md','paper_writing.md']:
 p=R/name;s=p.read_text();assert '## Current full-suite assessment — 20 October 2026'not in s
 s=s.replace('## Active completion request — 20 October 2026','## Historical completion request — 20 October 2026',1)
 first,rest=s.split('\n',1);p.write_text(first+'\n\n'+block+rest.lstrip('\n'))
print('Three instruction documents now reference the completed scoped assessment and remaining hardware/scientific limits.')
