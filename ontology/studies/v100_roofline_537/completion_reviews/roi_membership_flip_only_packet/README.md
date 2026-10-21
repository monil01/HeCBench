# Private flip-only four-model source-ROI packet

Original packet: `roi_membership_source_batch5_f/proposed_stage_patches.json`. This extraction preserves each flip stage patch, membership proof, selector/update and expected stage hash unchanged. Target original records equal the separately frozen original records exactly, with matching canonical record hashes. Current parent review hashes/indices were read under metadata LOCK_SH and rehashed before release; in this snapshot all four old/current parent hashes and indices are already equal, so **no rebase was necessary**. All original ledger, manifest, compressed plan/provenance/derivation bindings are unchanged and still current. Eight referenced source files match their old bindings. Detailed checks are in `refresh_review.json`.

Seven expected counts:

```
model_records: 4
names: 1
inclusion_declarations: 4
exclusions: 3
pure_alias_nulls: 0
inclusion_alias_nulls: 0
non_null_inclusion_stages: 4
```

Science: named FP32 4096-element contiguous3D16cube fixture, all three axes reversed. Native generic dimension/stride indexing and specialized Julia cubic mapping remain distinguished. Native/CUDA.jl/JACC separate warmup callsites are excluded; timed repeat callsites included. KA named family is included through its timed explicit launch callback. Same family also executes warmup, so actual sampled invocation remains unknown. No complete launch census, whole elapsedROI placement, extraFP64 regime, or causal claim is introduced.

The canonical accepted contract ID remains null and the historical result still lacks a configuration-ID field. This packet **does not pass the original generic null gate**. It requires root's separately reviewed exact-flip merge exception, the exact frozen config/result hashes, and an explicit root scientific note approving that narrow identity join. The exception candidate was approved only for forthcoming root dry-run, not apply/admission. No canonical field or historical artifact is edited here.

Root may combine this packet with batch6 only after its current batch5 application/audits finish and after independently checking all resulting initial parent hashes again. Preserve packet order in root approval plan. Parent/stage/selection/source drift must hold; no automatic inference or overwrite. No shared mutation, GPU/build/test, merge-engine execution, study import or Git action occurred. This packet includes private preparation source only, with no execution controller or reserved batch ID.
