# CMP: synthetic seismic semblance and stack scan

The complete four-stage source pipeline runs on two known synthetic SU inputs:
257 samples × 5 curvature candidates and 1,025 samples × 9 candidates. Each has
four gathers (hyperbolic Ricker pulse, constant, zero and opposite-sign traces),
with 16 and 52 retained traces respectively. An extra trace in each file checks
the original aperture rejection. Source windows are 3 and 15, within `MAX_W=16`.

The actual original SU reader, aperture filter, descending-trace-count gather
sort and linearization ran unchanged. An independent Python SU parser checks
every header field used by the kernels and every actual sample. Sort ties are
identified by complete group data; no stable ordering or ascending CDP ordering
is inferred. Both unchanged original CUDA CLI runs also pass its existing checker.
The large original DVC dataset is unavailable and is not claimed reproduced.

All four original GPU bodies are retained byte-for-byte in the extracted prefix
`native_kernels.cuh`; `kernel_extraction.json` binds it to the original full source.
The study changes the host harness, not the numerical operator. Source initialization
uses `(c1-c0)/nc`, so the candidate grid excludes `c1`. The source's signed scalar
handling, sample validity conditions, `-1` sentinels and first strictly positive
argmax are preserved. Julia uses C-style remainder for the negative sentinel.

## Scientific checks and scope

An independent Float64 oracle constructs interpolated trace-window matrices,
column sums, energy-normalized coherence and mean stack. It checks Cauchy bounds,
constant/zero/opposite controls and recovers the injected pulse-curvature indices
2 and 4. Their independent coherence scores exceed 0.99998, with gaps above 0.039
to the next candidate at the selected event times. This validates the named
synthetic operator, not a subsurface velocity model or application accuracy.

Predeclared floating acceptance is `abs(error) <= 5e-5 + 2e-4*abs(reference)`.
All candidate indices match the independent first-positive argmax exactly, and
also match a fresh reduction of the captured GPU score surfaces. Complete actual
external inputs, every staged sample buffer, guards and readonly arrays are checked.
Produced curvature/half-offset and score/stack arrays are captured internal inputs
under the ordered source pipeline; they are not falsely described as externally
fixed data that must be bit-identical before computation.

Computation and storage primarily use Float32. Unsuffixed source `FACTOR` and
`EPSILON` literals introduce Float64 source promotions, which the Julia code
retains; no compiled-instruction identity is asserted. The private source window
becomes an `MVector{16,Float32}`. Its compiled register/spill placement is unprofiled.
Saved artifacts encode Float32 and Int32 values losslessly as Float64.

## Measurement

Each case uses nine timed segments: initializer kernels, then four pairs of
sample H2D and scan/reduction segments. Transfers remain inside the measured
work as in the source. Whole-context completion, allocation, input preparation,
full readbacks and independent gates are explicit. Observations occur outside
the timer between segments, adding fences and cache effects relative to the
uninterrupted original CLI region. No original CLI wall-time claim is made.
All five trials are checked; only trial 1's time is discarded and 2–5 are averaged.
Host calls, source one-thread initializer groups and the two modest synthetic
sizes limit broader performance recommendations. No profiling/roofline claim.

## Clean-checkout regeneration

From this directory, using Python standard library and GNU C++17:

```sh
python3 regenerate.py --cxx g++
```

This regenerates the two SU files, executes original CPU input preparation and
constructs the independent oracle. Exact expected sizes/SHA256s are in
`data_regeneration.json`. Do not regenerate files while a verifier reads them.
The recorded platform uses little-endian GNU SU structure layout (240-byte
headers); arbitrary ABI or libm bit identity is not claimed. Hash mismatches must
be diagnosed, not silently accepted.

From the repository root, on the recorded pinned Julia/CUDA environment:

```sh
python3 tools/verify_portable_julia.py --config src/cmp-julia/study_config.json
python3 src/cmp-julia/audit_outputs.py --result /path/to/result.json --output audit.json
```

The central verifier owns the GPU6 lock. Do not wrap it in another lock. Explicit
configuration paths can be relocated in a local copy. Original five-trial results
are preserved separately; this clean config requests a fresh four-way run.

Per-model manifests, scientific intent, RQ1–5 observations, actual scoped token
windows and the frontend-rejected JACC candidate remain linked in this folder.
The rejected candidate needed whitespace in a generator expression; no device
compilation occurred before that repair. CPU oracle preparation separately
preserves a syntax error and an initially incorrect CDP-order assumption. These
are not native GPU defects or framework unsupported-feature claims.
