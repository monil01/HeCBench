# Five-unit LSTM recurrence

The historical Julia entrypoint aliases mcpr and does not execute the source LSTM. The restored code computes four gated affine transforms from the previous hidden state, then updates cell/hidden state and scalar projection at every timestep. Native device body is unchanged; only N8192->512 and T20000->257 dimensional macros differ in an explicit study header. The full original CLI is not certified.

## Fixtures and numerical evidence

Both supplied146-value weight files are identical (all0.01); all20000 supplied input values are0.001. The first two named cases retain these values, and cannot be counted as distinct parameter regimes. A separately declared third signed-dyadic fixture has8 distinct sequences repeated64times and varied gate/recurrent/projection weights. Fixture choice and tolerances precede target measurements. Every one of394752 timestep outputs and395190 exact input/weight-prefix values passes on all20 captures. Source/copy review binds canonical inputs to actual CuArray/native cudaMemcpy buffers. An independent Float64 scalar recurrence with math.fsum checks all outputs, maximum absolute error2.273e-9, under1e-5+1e-5*abs(expected), stronger than source1e-3. Float64 is oracle only. A reset-hidden/cell-state shortcut changes the later constant-input outputs; the audit records its exact failing-value count. This is an operator recurrence check, not trained-model or scientific forecast validation.

## Mapping, timing and effort

All four models map one entire sequence to a thread:2blocks of256, with serial257-step recurrence. Sequence-major layout keeps neighboring-thread loads strided257. Native private C arrays and Julia StaticArrays.MVector can compile differently; registers/spills/transcendental lowering are unprofiled, so observed time differences have no isolated framework attribution. No shared-memory communication, vendor computation or CUDA-specific device intrinsics. JACC explicitly requests zero dynamic shared bytes.

Each of five trials records the sum of three10-repeat kernel-loop means with blocking device-wide completion. Allocation, upload, oracle and complete capture stay outside. Discard only timing1; preserve all retained samples. This three-case sum differs from the original two-case average and is identical across all models.

Current CUDA.jl restoration has2 program/process candidates: one shared host-driver parser rejection before JIT, then the first actual device candidate passes. KA/JACC each pass their first wrapper/device check with the already verified newly authored shared mathematical helper. That helper belongs to reference-restoration effort once. Three numerical cases and five trials are not compiler attempts. Historical counts/costs are unknown. Prospective counter windows include cached/repeated context, paper publication and team/document coordination; workflow overlaps subphases, is nonadditive, and is not exclusive billed translation cost.

Two shared postverification Python annotation errors (conditional-token spacing and usage-record key) were repaired using unchanged saved artifacts, with no GPU rerun. Failed script versions were not archived before editing; diagnostics remain in tool history. These are annotation T1 events, not device compiler attempts.
