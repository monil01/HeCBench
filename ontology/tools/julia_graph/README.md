# Julia relationship graph

Run from the repository root:

```sh
python3 ontology/tools/generate_relationship_graph.py
```

The generator writes `ontology/reports/relationship_graph.html` and `ontology/relationships/julia_relationships.json`. The HTML contains all data and code and works offline. `--root PATH`, `--output FILE` and `--neighbors N` are available; the default sparsification retains the union of each endpoint's top five positive links. The JSON index is generated under the selected root even when a different HTML output path is supplied.

Inputs are the scientific-intent and per-model implementation JSON files in `ontology/studies/julia_abstractions_447/`. The optional reviewed-class reconciliation is read from `2026-PASC-Hecbench-Julia-paper/research/revision_experiments/classes/normalized_records.json`. If absent, the graph uses recorded class declarations, marks their unreconciled basis, and leaves device API dependence unknown. Original catalogue records and `similarity_edges.jsonl` are preserved.

The page provides four views:

- **Scientific operators:** reviewed family equality and lexical overlap of scientific-task and mathematical-method descriptions.
- **Julia implementation properties:** relationships within CUDA.jl, KA or JACC, using recorded class, API dependence, algorithm, layout, mapping and storage precision.
- **Verification and feedback:** actual failed events whose recorded origin matches the selected model, agreement-kind descriptors and qualified outcomes. Accepted drift/adaptations and shared/native-origin failures are not silently attributed to a Julia frontend.
- **Native CUDA → Julia versions:** the three recorded Julia versions of a selected benchmark and their configuration-specific reference/acceptance qualifications. Hardware barriers, property checks and execution-only outcomes remain distinct from numerical agreement.

Every retained edge contains the matched properties, shared values or lexical terms, fixed weights and contributions. Source records and reviewed-class reconciliation are linked from the evidence panel. Node data binds scientific and implementation records by SHA256. The metadata identifies the source fingerprint, generation time, class input and per-view/model edge counts.

The fixed weights are author choices for exploration. They are not calibrated probabilities, semantic-equivalence proofs, embedding distances, causal performance explanations or a target-selection predictor. Missing values add no score. Scores use a fixed denominator, so missing fields can lower scores. A missing edge can result from sparsification or incomplete metadata; it does not demonstrate dissimilarity. Ties are broken by neighbor ID. Filtering operates on the saved sparse graph and does not recompute nearest neighbors after excluding names.

The default neighborhood keeps the selected benchmark and up to twelve strongest retained eligible neighbors, including connections between those neighbors. Whole-study mode keeps every matching node, including isolated nodes. Colors identify scientific families, implementation classes or scoped outcomes according to the selected view. Translation-view colors identify CUDA/CUDA.jl/KA/JACC. All visible edges are also accessible as buttons, and the selected graph can be exported as JSON. The exported nodes preserve per-model provenance and exact recorded configurations.

The inspector displays separate model records, exact configuration trial verdicts/times, qualified references, environment defaults and overrides, source reviews, failure history, attempts and token-accounting scope. Configuration means remain the recorded means of trials 2–5. No GPU work or new measurement is performed by this report. Token windows may overlap; saved-source token estimates remain proxies.

## Live full-suite view

Generate the current CUDA catalogue view with `--population 537`. The generators retain `--population 447` for the frozen assessment. The live summary reads the source/hash-bound selected-contract census and keeps numerical, property, execution and hardware outcomes separate. The explorer exposes the additional word2vec asynchronous property workflow without adding a second benchmark name.
