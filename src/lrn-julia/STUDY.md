# LRN: full forward and gradient, corrected reference

## Source operator and defects

The intended operator is cross-channel local response normalization in NCDHW storage: y_i=x_i*(1+alpha/5*sum_window x_j²)^(-.75), with its vector-Jacobian backward gradient. Sourcewindow5, alpha=.000122, beta=.75 and k1 are retained, including divisor5 at truncated channel edges.

The unchanged native header omits depth in both address functions and initializes the backward omega accumulator as integer. A direct unchanged-kernel diagnostic has0accepted/1rejected full9520-value capture and6484 values outside the predeclared oracle budget. At every workgroup, the5Dcase has769/840 wrong forward and780/840 wrong backward values. The2Dcontrol has correctforward but72/84 wronggradients(maxabs1.37037e−5), separating the backward integer-accumulation issue from depth addressing. Guards remain exact. Static depth-aliasing duplicate stores are source evidence; no race sanitizer was run.

The diagnostic zero-initializes outputs. This agrees with originalForward but differs from originalBackward's initialdiff_src=src in depth regions the defective kernel never writes. Thus6484 is the declared diagnostic count, not a run of the huge originalCLI. The fully written2Dbackward counterexample is independent of that initialization. OriginalBackward also reads back upstreamdst rather than actualdiff_src; its printedchecksum is not gradient validation. Hash-bound study_adjudications.json clarifies these scopes without changing any run.

## Explicit corrected-reference secondary

native_corrected/kernels.h changes only two NCDHW address formulas and the backward accumulator toFloat32. Original source is unchanged and remains blocked. Native_correction_provenance.json binds original/corrected/patch/actualwitness hashes. The restored arithmetic and unique addresses change the defective reference, so these timings are not untouched-native results. Other inactive layout/spatial branches are outside the contract.

Historical Julia already implements genuine NCDHW addressing and Float32 accumulation. Every existing function is byte-preserved; only the entry/harness changes. This is native reference correction, not a Julia semantic repair. Historical checksum PASS did not provide modern full-output evidence.

## Numerical and scientific evidence

Named shapes2×7×3×4×5 and1×4×1×3×7, positive source-like and signed independent-upstream fixtures, all64/128/256/512 workgroups, R10. Huge original shapes are unmeasured. Independent Float64 forward and analytic gradient are cross-checked by central finite differences of weighted scalar forward loss for all924 coordinates. This validates the continuous LRN operator, not differentiation of the quantized Float32 program or whole-model training accuracy.

Every model runs five checked in-process trials after explicit full warmup. Mean trials2–5 retains all outliers. Each capture has1872 actual input-bit/parameter prefix values,7392 full forward/gradient values for16 components, and256 exact device guards. All20 accepted artifacts are bit-identical across the four models, within predeclared8e−7+4e−6*absoracle. Maximum absolute oracle error is9.2761192e−8. saved_output_audit.json independently verifies all values, metrics, guards and hashes; component_timings.json retains all component means and outer sample variation.

## Mapping, runtime and effort

All three Julia versions are framework_device_kernel with two definitions: independent channel-window forward and nested-window backward gathers. KA uses framework indexing; JACC receives the exact padded grid stride from the host, avoiding foreign device intrinsics. There are no atomics/shared allocations; JACCshmem0 explicit. Total mean microseconds(sum16componentR10loop means):CUDA105.069150, CUDA.jl128.486150, KA352.018275, JACC169.531700. Allsamples are retained, with no guessed explanation of outliers. Grid-object/spec construction, host launch dispatch and index/codegeneration are covariates. Source printed2N/3N byte estimates ignore neighbor reads and are not measured traffic or roofline coordinates.

Existing Julia/reference and both target programs passed firstprocess/JITchecks, targetonecandidate/twodefinitions each/zero feedback repairs. Originalnative diagnostic and correctednative builds count separately (two builds/one composite native repair). Kernel-body preservation, reference instrumentation, independent oracle and native repair are separate cost phases. Historical attempts/tokens unknown; saved-source proxies/mechanical reuse and context/cache-inclusive overlapping prospectiveusage are not exclusive billed costs.

## Reproduction

study_config.json selects final20261018T075812211452Z_lrn correctedsecondary. Originaldiagnostic/ref/KA configs preserve provenance and exact unchanged-series reuse. generate_oracle.py/fixture_provenance.json reproduce four essential CPU input/oracle binaries and the all-coordinate derivative checks. reference_original retains historical Julia; native_corrected and native_lrn_correction.patch retain the minimal intervention. GPU diagnostic outputs stay local by hash. Canonical guides and worker checkpoint resume the queue; root alone commits verified kernels separately.
