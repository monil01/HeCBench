# Scientific and implementation ontology: frozen300 census

This study layer extends the frozen100 pilot without rewriting its evidence.

- `domains/<benchmark>.json`: one scientific-intent record for every frozen name. Provisional groups organize review; they are not validated application provenance or measured bottlenecks. Domain-neutral primitives and hardware probes remain explicit.
- `implementations/<benchmark>/{cuda_jl,ka,jacc}.json`: 900 distinct implementation slots. Every slot records honest status, source syntax/hash evidence, per-model source-review status, exact recorded configuration links, effort fields and observed failure events. Null ontology fields mean not yet reviewed. Legacy shared traits do not certify individual implementations.
- `failure_taxonomy.json`: stable event categories and implementation/phase/version dimensions. Counts of events, affected programs, repairs and final failures are different quantities.
- `failure_events.json`: only explicitly coded observed events; partial histories do not imply zero failures.
- `index.json`: current population/status/review inventory, not a claim that300 ports are verified.
- `PROGRESS.json`: small combined source/paper checkpoint pointer. Full checkpoint/data remain local and excluded from Overleaf.

Generate with `python3 tools/export_julia_300_registry.py`; annotations are read from `.porting-state/portable_julia/traits/<benchmark>.json` and `scientific_domains_300.json`. Scientific annotations need source evidence and confidence; syntax-pattern extraction alone cannot validate algorithm fidelity, scientific provenance or a bottleneck. Runtime release and implementation model are separate axes. Measured trial bundles remain the authority for exact environment and numerical/performance claims.

A community-facing artifact should let scientists find comparable mathematical operators and numerical requirements, inspect how each implementation realizes them, reproduce scoped accuracy/performance evidence, and identify unanswered portability questions. This source layer is not a claim that local research data have been publicly archived.
