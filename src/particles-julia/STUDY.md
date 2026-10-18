# Particles: full source pipeline, one-step certificate

## Scientific contract

This source is a soft-sphere particle/contact demonstration under gravity, box reflection and a fixed spherical cursor. It is not a Lennard–Jones/mutual-gravity model or a validated material/conservation solver. OriginalN=16384, grid64³, seed1973 jittered initialization, Float32 parameters and unusual worldOrigin.x=+1 are preserved. Contact force is added to velocity directly, without a dt/mass multiplier, exactly as the source does.

The primary named contract performs **one complete physical update**, retaining all8 active generic kernel definitions and26 launches: integration,hash,21-launch descending bitonic sort with source tie swaps,cell clear/bounds/reorder,and27-cell contact update. All stages execute on GPU. Original Makefile10000-step trajectory is **unverified**. The fixed cursor is geometrically unreachable from source bounds, so its branch is retained but not numerically covered.

## Evidence and timing

Final bundle: `20261017T235957743435Z_particles`; config `study_config.json`. Every model has five warmed in-process trials, mean runs2–5. Timed region includes the full source-equivalent host dispatch loop and blocking device-wide completion. Reset/transfers/CPU oracle/serialization are outside. No retiming or fastest selection.

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
|126.971µs|131.094µs|266.320µs|439.546µs|

All20 primary artifacts pass independent CPU-only revalidation (`audit_saved_outputs.py`, NumPy2.5.4),950300Double values each, including131100exact input-prefix values. Outputs include full particle position/velocity,hash/index,cell start/end and reordered position/precollision velocity. Integer metadata remains UInt32 onGPU; Double serialization represents it exactly. EmptycellEnd is unused in source and canonicalized sentinel for serialization only. One-step independent Float64allpairs physics maxerror1.80334e-8 < predeclared absolute1e-5; full sort/permutation/cell properties are exact. Raw source kernels untouched; unread maxParticlesPerCell and overwritten staging initialized only to define the study harness.

CUDA.jl uses own CUDA.jl staticshared/barriers. **KA is hybrid_device_intrinsics** because reorder retains one CUDA.blockDim().x workgroup-width query; all other indices/shared/barriers are KA. This is disclosed backend coupling, not evidence KA lacks the operation. JACC uses own indices/shared/sync and explicit0/4096/260B per-stage LaunchSpecs. JACC shared-template copies/barriers and framework dispatch may differ; no unmeasured bandwidth/occupancy/causal performance attribution.

## Failures and diagnostic limits

Historical Julia only performed unbounded gravity integration. Its actual same-fixture GPU witness (source body preserved verbatim) has maxstateerror.000303924 and1031particles outside the source cube, missing horizontal contact velocity. `reference_original/main.jl` and `diagnostics/historical_*` preserve proof; full restoration effort is separately recorded.

The **eight-step native trajectory fails** the original declared1e-5 independent Float64 oracle: maximum2.62650125704e-5. No tolerance was relaxed and this is not part of certified results. Original five failed-run structured logs remain, but old runner discarded their binary artifacts. After root fixed diagnostic retention, one distinct **untimed** eight-step diagnostic preserved all8fullstates as one7,602,400-value artifact. It remains rejected:0accepted trials,mean null,1rejected record. `diagnostics/rejected_capture_provenance.json` binds counts/result/hash. Finite+nulltime capture was exercised; nonfinite-string handling was not.

CPU-only localization from saved states finds exact hash/permutation/cell/reordering constraints at all8steps. Double integration from each actual previous Float32 state changes positions about3e-8 yet can change one-step velocity3.23e-5. Independent all-pairs force evaluation on actual GPU integrated positions/prevelocity differs at most6.46e-6. This supports sensitivity of near-contact normalization to integration rounding, rather than an observed missing-cell interaction, but does not prove a sole cause or certify the eight-step trajectory. Saved state, analysis source and per-step residuals remain available. No native/scientific arithmetic correction was made.

## Effort and restart

One restored CUDA.jl candidate/device-JIT process check,one KA candidate/check,one JACC candidate/check;8generic definitions each,0target feedback repairs. Historical GPU diagnostic and native/CPU oracle builds are separate. Internal compiler passes/specializations unknown. Shared helper/math cost belongs current reference phase. Prospective windows include context/cache/reasoning/coordination and overlap the broad workflow; they are not additive exclusive billed costs. Source chars/4 proxies are separately partial. Reference/KA/JACC phase outcomes, manifests, traits, failures and usage boundaries are checkpointed. Root coordinates separate verified-variant commits. Resume remaining queue atpns after paper/checkpoint registration.
