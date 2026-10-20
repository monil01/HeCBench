# MCMD: full source molecular-dynamics trajectory

## Native readiness and scientific scope

The original CUDA source and both original dataset files are preserved byte-for-byte in `reference_history` with hashes in `source_archive.json`. Its unmodified CLI built and executed the original 10 fs water/octane NVT Nose–Hoover trajectory. The PQR has 3,379 ATOM records, including eight BOX markers skipped by the original parser: 3,371 physical atoms and 955 movable molecules are simulated.

The study keeps the original setup, random seed, all CPU integration/rotation/thermostat/potential/observables and file-I/O behavior. Its selected GPU path is Lennard–Jones plus real/reciprocal Ewald electrostatics, kmax3, 11 force evaluations and 11 complete molecular/atomic state observations. Other CLI modes and the polarization branch are not certified by this fixture. The original source's full GPU force body remains unchanged in its native reference; the adapter exposes the exact device inputs and outputs and a non-null target callback cannot fall back to the native kernel.

The predeclared artifact budget is absolute2e-6 plus relative2e-10 for all Float64 fields. The initial actual-device input prefix is 47,220 Float64 values, compared bit-exactly. Every force evaluation is independently checked against long-double all-pairs LJ/real-space forces and mathematically factorized reciprocal structure factors. This oracle does not call either source CPU or GPU force functions. Every non-force atom byte must remain unchanged. Complete molecular/atomic state is finite; rigid intramolecular squared distances and normalized total-force balance have explicit checks. These are tested engineering budgets, not universal physical accuracy or thermal-equilibrium guarantees. The common original CPU integrator has not been independently rewritten.

`oracle_controls.cpp` separately checks the force oracle against a five-point gradient of pair energy, whole-cell periodic translation, equal-and-opposite force, intramolecular exclusion and frozen-pair exclusion. A preserved initial control script forgot to restore molecule0's ID after its exclusion test; the control-fixture correction is separate from GPU or oracle mathematics.

## Measurement

A complete pipeline warmup precedes five checked in-process trials; report the mean of trials2–5. The timed region includes the full source CLI setup, host computation, file I/O and original force allocation/copy/compute/free. Added actual-input readbacks, independent oracle work, complete-state study observations and artifact serialization are excluded identically. This is not a kernel-only time. Original device allocations have no synthetic guard regions; no full memory-safety proof is claimed.

## Reproduce CPU oracle controls

```sh
g++ -std=c++17 -O2 src/mcmd-julia/oracle_controls.cpp -o /tmp/mcmd_oracle_controls
/tmp/mcmd_oracle_controls
python3 src/mcmd-julia/prepare_controller.py
```

Reproduce all four complete paths with the existing pinned Julia/CUDA environment:

```sh
python3 src/mcmd-julia/make_config.py --models all --output /tmp/mcmd_study.json
python3 tools/verify_portable_julia.py --config /tmp/mcmd_study.json
# Equivalent entrypoint:
make -C src/mcmd-julia run
```

The central verifier alone owns the shared GPU6 lock. Do not add an outer lock around it. The full original dataset is tracked in the source archive; no external assets or OS package changes are required. Both references are automatically included for individual KA/JACC runs.

The callback adapter retains the source force body, input/output representation and launch geometry. Its independent allocations are issued before their host-to-device copies rather than interleaved as in the original wrapper; this common host-adapter ordering is shared by all measured models. The original unadapted CLI remains a separate execution diagnostic. Source-printed `Device offload time` includes study oracle overhead and is not the comparative performance metric. Only structured trial ROI values subtract the added observations/checks. The reported `max_abs_error` is the computed all-force oracle error; `max_rel_error` is null because no independent relative maximum is computed. The verifier separately computes full-artifact dual-reference errors.

## Accepted accuracy, runtime and scope

All four paths passed five trials. Each artifact contains2,517,798 Float64-encoded values, including the47,220-value exact initial actual-device input prefix. `all4_full_capture_audit.json` independently rehashes/parses all20 captures and checks every returned atom's non-force bytes, each selected state and rigid intramolecular distances. `all4_accuracy_timing_summary.json` preserves every trial and the separately computed long-double force errors.

| Model | Mean of trials2–5, seconds |
|---|---:|
| CUDA |4.9138202075|
| CUDA.jl |13.748848139|
| KA |13.143543559|
| JACC |13.1321114315|

CUDA.jl's retained trial3=15.582581117s was not dropped. All three Julia dispatches share the same authored selected force equations, CUDA Float64 atomics/libdevice erfc/rint, and native CPU controller. Native-vs-Julia differences include compiler/source mappings; indexed Julia AoS field loads replace the native anchor POD copy. No profile/control establishes which mechanism caused the slowdown.

The captured host state is explicitly: twelve molecular three-vectors (force, torque, COM, original COM, diffusion correction, original velocity, velocity, old velocity, angular velocity/acceleration/old acceleration/position), seven atom three-vectors (position, force, velocity, old/original velocity, torque, dipole), mass/parameters and four source statistics (potential, kinetic, total energy, temperature), plus thermostat state. Statistics follow the original update cadence. Other host statistics and complete CLI file/text bytes are outside agreement. Full source CLI work still executes.

Post-hoc descriptive physical-field comparison against saved native captures (`physical_field_accuracy.json`) finds maximum position difference7.1055e-15Å, molecular velocity difference5.9848e-17Å/fs, returned force difference4.166e-10K/Å, and temperature difference1.137e-13K over all Julia models/trials. These observations do not change the predeclared acceptance budget or certify equilibrium chemistry. The independent long-double force oracle's maximum error across all four paths is8.485e-10K/Å; normalized total-force balance is below2.57e-16.

```sh
python3 src/mcmd-julia/audit_captures.py --result RESULT_JSON --output /tmp/mcmd_captures.json
python3 src/mcmd-julia/analyze_accuracy.py --result RESULT_JSON --output /tmp/mcmd_physical_errors.json
```

The accepted all-model receipt is `.porting-state/portable_julia/runs/20261020T022748162082Z_mcmd/result.json`. The source original input SHA256 values are recorded in `source_archive.json`; all measured includes/assets are explicit dependencies with full-file bindings. The new parent `main.jl`/Makefile/run helper are unmeasured integration entrypoints; measured nested entries and source dependencies remain unchanged.
