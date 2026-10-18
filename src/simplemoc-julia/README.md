# SimpleMOC: conditional corrected single-segment transport moments

The [official SimpleMOC-kernel project](https://github.com/ANL-CESAR/SimpleMOC-kernel) identifies neutron-flux attenuation across a geometric segment as a reactor-physics MOC proxy component. This study verifies a bounded attenuation operator, not a full reactor solver, source reconstruction, geometry, criticality calculation or concurrent multi-segment simulation.

## Independent failures and scoped correction

The original CPU checker and GPU routine share two moment defects. The unchanged constant-source control passes. For an affine positive source with sigma1, outgoing flux agrees with independent quadrature, but the GPU flux integral is .1078585014 instead of .0991438633. Its linear-source integral term lacks a factor1/2. For a positive quadratic fitted source with minimum about .1 and sigma2, the original GPU outgoing flux is -.1578246206 instead of about +.06267535. The reused quadratic moment divides only its exponential term by sigma cubed instead of the entire numerator.

The original CPU/GPU programs, all14 scratch fields, flux/state values and exact two-expression patch are retained. CPU witnesses use4096-node long-double Simpson quadrature; actual x86_64 long double has a64-bit significand and16-byte storage, not universal128-bit precision. GPU witness bodies are byte-identical extracts apart from include paths. Minimal witness input equality was checked on actual readbacks; their numeric input echo arrays were not serialized, and no five-trial performance claim is made for those diagnostics.

The conditional kernel places the whole reused moment numerator over sigma cubed and halves its linear flux-integral contribution. Canonical native_reference_repaired/provenance were declared before the first trial. These timings are excluded from untouched-native CUDA aggregates. The original failures remain failures.

The declared attenuation equation is dpsi/ds+sigma*psi=q0+q1*mu*s+q2*mu2*s². Independent convolution moments H0=(1-exp(-sigmaL))/sigma, H1=(L-H0)/sigma and H2=(L²-2H1)/sigma give outgoing flux and its integral. The integral balance sigma*J=integral(q)+psi_in-psi_out provides a separate control. Original fitted coefficients and placeholder constants dz.1,zin.3,mu.9,mu2.3,ds.7,weight.5 are retained. The interior axial derivative uses the original reversed neighbor convention; its geometric interpretation is ambiguous and is not silently repaired. This verifies attenuation of the fitted polynomial, not the axial discretization's physical accuracy.

## Complete numerical evidence and limits

Six fixtures use two source regions, five fine axial intervals and1,17,128,33,65,16 energy groups. They cover first/interior/last intervals, constant and signed-affine/quadratic positive sources, zero-source absorption and constant-source equilibrium. Each case has one active segment as in the source's documented VERIFY mode, with1,2,3,2,3,2 repeated attenuations. Cross sections are1–2, optical depths .7–1.4 and repeats at most3; optically thin/small-cross-section stability is not certified.

An independent Decimal70 oracle uses exact Float32 inputs/constants and integration-by-parts moment recurrence rather than the native expanded expressions. A read-only audit additionally integrates the saved fitted polynomial directly by Simpson quadrature at128/256 nodes. All20 complete19187-word artifacts pass:6023 exact prefix words plus13164 full-state words, including all FSR fluxes, outgoing state,14 global scratch fields, actual source/cross-section/IDs, actual initial mutable-buffer echoes and672 guards. Inputs, all guards and untargeted FSR values are checked exactly. Nonnegative transport/tallies, absorption and equilibrium controls pass.

The predeclared engineering budget is2e-5+2e-5*abs(reference), not a formal roundoff bound. Maximum Decimal-oracle absolute error is1.99e-7 and maximum budget fraction .00829. Direct Simpson error is1.26e-7,128/256 quadrature difference1.61e-10, and moment-balance residual2.14e-7. Balance and Simpson use an independently predicted pre-last state; actual intermediate GPU trajectories are not captured.

Small absolute errors do not imply uniform relative accuracy. Cancellation-scale q1/t2 components have relative errors about3.56 despite absolute errors1.66e-7/2.93e-8; outgoing-state and integral maximum relative errors are4.11e-7/3.04e-7. Per-field nonzero-relative, zero-reference absolute and normalized-budget metrics remain in output_audit.json. No tolerance changed after a failure.

## Mapping and performance scope

The original native launch for one segment uses128 blocks by128 threads. All four versions retain this overlaunch, with exactly one active work item serially looping over energy groups. All14 scratch planes remain global output arrays. Historical genuine Julia used private temporaries and inherited the original moment formulas; it is archived intact. The study's shared corrected numerical helper is authored once in the reference phase and reused by CUDA.jl, KA and JACC dispatch wrappers.

KA and JACC use their own global indices and dispatch with ordinary Julia Float32 arithmetic/exp, without CUDA-specific device intrinsics. They need no shared memory, workgroup barriers or atomics; JACC explicitly requests zero shared bytes. Kernel/LaunchSpec objects are cached outside timing. Native default fusion and Julia code generation can differ; machine instructions/resources are unprofiled. Only CUDA backends have been tested.

The source's larger concurrent path shares one global track-state vector and one14*G scratch buffer among segment threads, with non-atomic FSR additions. That static ownership hazard is outside certification; no bad multi-segment GPU run is invented. This verified one-segment mode cannot establish correctness or performance of the default millions-of-segments CLI.

Each trial sums six kernel regions totaling13 launches plus per-case completion. Reset, allocation, input setup, complete readbacks, scientific checks, serialization and explicit all-case warmup are outside. Five checked trials remain and timing1 alone is ignored;2–5 are averaged without outlier removal. A mostly inactive grid and one serialized energy loop are not representative of production MOC scaling or a basis for generic roofline/framework recommendations.

Current corrected CUDA.jl/KA/JACC candidates each compile and pass their first check. Original scientific failures are separate native origins, not framework compiler failures. Shared numeric helper, correction, witnesses and oracle effort is attributed once and reused KA then JACC. Prospective windows include repeated/cached context and coordination, overlap the outer workflow and must not be summed or called exclusive billed costs. Marginal ordered dispatch-wrapper effort is not an intrinsic cold-start framework translation cost; historical creation cost remains unknown.
