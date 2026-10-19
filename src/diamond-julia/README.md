# DIAMOND: qualified Tantan HMM masking component

This ports the fixed-33-letter repeat-masking GPU component, not the complete DIAMOND aligner. Full short original CLI was built/executed; its CPU-before-GPU masking and incorrect 64-stride table prefix are documented in `host_layout_witness.json`. The primary reference uses proper compact likelihood packing, unmasked input and an explicit terminal inactive-lookup bounds guard; original CUDA is unchanged and old captures retained. Excluded from untouched-native aggregates.

## Reproduce

From the repository root (Python 3.12.3, standard library; pinned Julia projects in `environment_snapshot.json`):

```sh
python3 src/diamond-julia/generate_assets.py
python3 src/diamond-julia/prepare_cold_config.py /tmp/diamond_cold.json
python3 tools/verify_portable_julia.py diamond --config /tmp/diamond_cold.json
```

The generator uses the committed original BLOSUM62 score source and independently expressed 51-state HMM. All essential CPU binary assets are exactly regenerable; saved GPU captures remain local, with hashes/provenance in the audit. `audit_saved_outputs.py` audits the original local bundle and cold regeneration without GPU work. Do not overwrite measured assets or use stale absolute configs for a relocated checkout; the cold helper rewrites paths/removes reuse.

## Contract

386 strings in batches 1/128/257, 33 letters, max repeat distance 50, default no-gap parameters and threshold .5. Each of five real trials captures 26,873 Float64-encoded values, including a 14,039-value exact actual-device input prefix; raw mask bytes and guards are exact. The resident three-launch ROI excludes captures and original alignment/CPU work. An additional actual untimed GPU helper observer captures 12,738 posteriors per model against independent HMM tolerance 3e-7 + 3e-7*abs(reference), with exact flags/guards. It does not claim access to timed intermediate arrays.

Mixed Float64 recurrence/private work and Float32 posterior rounding are preserved. All target device kernels use their own framework APIs and private StaticArrays; only the CUDA backend is tested. Other sequence lengths, gap modes, arbitrary thresholds, full alignment quality and production size scaling remain unverified. Read `paper_observations.md`, `native_readiness.json`, `study_adjudications.json` and `scientific_witness.json` for qualifications and retained diagnostics.
