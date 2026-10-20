# Ontology report generator

Regenerate the self-contained explorer from the repository root:

```sh
python3 ontology/tools/generate_ontology_report.py
```

Python requires PyYAML. `--root PATH` selects another source checkout and `--output FILE` selects another destination. Open `ontology/reports/ontology_report.html` in a current Chrome, Edge, Firefox or Safari browser. The embedded study data uses the browser's native gzip decompression; no server, CDN or external request is needed.

Inputs:

- `ontology/benchmarks/*.yaml`: complete catalogue records, including inferred annotations.
- `ontology/studies/julia_abstractions_447/domains/*.json`: reviewed scientific intent and witnesses.
- `ontology/studies/julia_abstractions_447/implementations/*/{cuda_jl,ka,jacc}.json`: separate implementation, verification, configuration, performance, effort and provenance records.
- Optional `2026-PASC-Hecbench-Julia-paper/research/revision_experiments/classes/normalized_records.json`: reviewed class reconciliation and authored-kernel count observations. If absent, the reconciled class remains unknown; the recorded implementation class is still available. This input is identified in the embedded report metadata.

The generator does not run benchmarks, reclassify sources or modify ontology records. There are 85 initial columns covering identity plus twelve property categories. The property search exposes every nested scalar path in the source records; **Show as column** adds any such field to the table. Raw array-of-object paths use `[]`, while scalar-array values share one field.

Each explorer row belongs to one benchmark/model. Catalogue-only names receive one row without an invented Julia outcome. Filters combine with AND; individual array fields match any member. Filters on different configuration fields can match different configurations of that same implementation, so inspect **Exact recorded configurations** before drawing a configuration-specific conclusion. An empty filter result is shown explicitly. Unknown and missing values are searchable; false and zero remain real recorded values. A numeric inequality never converts a missing value or boolean into zero.

The table supports column/category selection, a compact preset, numeric/text sorting, pagination, combined filters, global search, a record inspector, and CSV export of all matching rows and visible columns. Catalogue charts retain benchmark-level counts, and chart clicks filter the explorer. The reviewed scientific-family chart uses the 447-name study denominator. The 538-record catalogue includes names outside the study.

Configuration means remain the recorded arithmetic means of trials 2–5. They are not averaged across configurations here. Token windows can overlap, and saved-source token estimates remain explicitly labeled proxies. A configuration's outcome may reproduce a known native defect or use a corrected reference; the status and agreement fields retain those qualifications.

## Live full-suite view

Generate the current CUDA catalogue view with `--population 537`. The generators retain `--population 447` for the frozen assessment. The live summary reads the source/hash-bound selected-contract census and keeps numerical, property, execution and hardware outcomes separate. The explorer exposes the additional word2vec asynchronous property workflow without adding a second benchmark name.
