# TestSNAP: complete source force pipeline

This study preserves the seven original GPU kernels and original J14/2000-local-atom/26-neighbor/100-force-evaluation workload. Additional supplied J2 and J4 cases use three evaluations. It does not integrate an MD trajectory or establish material-model accuracy.

## Scientific checks and qualification

The independent C++ oracle uses explicit SU(2) symmetric-power polynomials, three-coordinate long-double dual derivatives, and reverse differentiation of the complete Clebsch–Gordan tensor energy. This differs from the source recursive U, analytic DU and Y algorithms. It checks all pair gradients and local/ghost forces. J14 and J2 also match every supplied author force component. J2/J4 independently pass five-point energy-gradient differences for all 48 coordinates. Full J14 finite differences were not computed. Long double is the platform extended type, not a claim of IEEE Float128.

The supplied J4 author force field has 105 exact zeros. Its strict author-field auxiliary check failed; the independent operator has nonzero forces. The failed criterion and observations remain in `candidate_history/acceptance_v1/` and `J4_supplied_reference_review.json`. The declared revised contract uses the strict independent gradient and finite-difference checks for J4 while still reporting its mismatch to that supplied field. This is a revised acceptance-contract stratum, excluded from ordinary untouched-native aggregates. No native kernel/scientific correction is claimed. J14/J2 criteria and the 1e-7 + 1e-9*abs(reference) full numerical budget are unchanged.

An initial oracle error-metric overload bug was caught before acceptance. The known reconstruction and missing original snapshot/log are explicitly documented in `candidate_history/oracle_metric01/metric_history.json`; do not treat reconstructed history as original retained bytes.

## Timing and evidence

Each version performs one complete untimed warmup, then five checked trials. Mean performance excludes the first trial and uses trials 2–5. The measured region sums all 100+3+3 source step pipelines: force reset, repeated source AoS→SoA geometry/neighbor/weight/cutoff preparation, three H2D copies, seven kernels with four source completion points, gradient D2H, own host force assembly and author-error accumulation. Input/guard capture and independent checks pause the clock. An explicit additional common context wait after H2D, defined zero workspaces, guarded allocation and observer gaps are instrumentation/mapping covariates. These timings are not original uninstrumented CLI end-to-end or pure GPU-kernel timings.

Every timed step retains all pair gradients and all local/ghost force components, not only RMS. Each trial has 17,715,669 Float64 fields, with 632,233 exact serialized input-prefix values. Separate lossless actual-input CAS ledgers cover repeated input readbacks, and raw sidecars contain actual 512-byte guards for each of 16 buffers at every step. The CPU-only `study_observer.cpp` checks/captures values; it implements no GPU kernel or scientific force driver. Julia owns its host transpose and force accumulation.

## Assets and preparation

The source-declared DVC archive was recovered with its exact MD5 `a1658855b569a2ba8ae42d1b677c182e` and 2,873,476-byte size. URL, SHA256 and extracted identities are recorded in `reference_asset_retrieval.json`. The three used original headers are retained under `reference_headers/`; J8 is not measured. `data/fixture_manifest.json` and `data/independent_oracle_manifest.json` bind all prepared binary inputs and oracle outputs.

CPU fixture export uses unchanged source index/CG setup formulae; zero allocation defines inactive exported table bytes. The independent oracle writes pair-gradient/force data from the source fixture. Reference headers, six small source/oracle fixture binaries, textual manifests and generators are essential checkout assets. Large raw output/CAS captures, executables and `.so` files are research artifacts outside source Git.

Example preparation (unprivileged; run from this directory):

```sh
for J in 14 4 2; do
  g++ -O3 -std=c++17 -DREFDATA_TWOJ=$J -Ireference_headers prepare_fixture.cpp -o /tmp/snap_fixture_$J
  /tmp/snap_fixture_$J data/J${J}_source_fixture.bin
done
g++ -O3 -std=c++17 -fopenmp independent_force_oracle.cpp -o /tmp/snap_oracle
OMP_NUM_THREADS=16 /tmp/snap_oracle data/J14_source_fixture.bin data/J14_independent_oracle.bin
OMP_NUM_THREADS=16 /tmp/snap_oracle data/J2_source_fixture.bin data/J2_independent_oracle.bin
OMP_NUM_THREADS=16 /tmp/snap_oracle data/J4_source_fixture.bin data/J4_independent_oracle.bin --J4-unvalidated-zero-author-field
```

`translate_source.py` requires an existing user-space pycparser and system `cpp`; its output is a mechanical source mapping, not the independent oracle. It preserves lexical scopes, Int32 divisions/indexing and flattened AoS complex layout. `cpu_translation_probe.jl` is an additional small-case source-mapping diagnostic, not GPU evidence.

All four versions passed the final receipt `.porting-state/portable_julia/runs/20261020T050030123952Z_testSNAP/result.json`. `final_source_artifact_audit.json` independently rehashes all 20 primary artifacts and source bindings. Per-model `*_capture_audit.json` files additionally check actual input CAS, complete physical outputs and raw guards. Exact effort units and qualifications are in `paper_observations.json` and target accounting. Native C++ versus Julia host work is part of the pipeline comparison; no abstraction-only causal claim. Shared CG coefficients are declared inputs, not independently regenerated physics parameters.
