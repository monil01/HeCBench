# FPDC: lossless packet validation and corrected reference

## Scientific operator and original counterexample

The source GFC-style compressor predicts each binary64 bit pattern from a prior 32-value group, encodes unsigned differences with sign/byte-count nibbles, and packs little-endian payload bytes. This is integer coding of opaque IEEE64 representations, including signed zero, infinities and NaN payloads. Losslessness requires recovering every original bit.

The actual unchanged HeCBench native diagnostic has 1,218 wrong payload bytes and fails all eight independent decoder round trips, although all eight output offsets/sizes are exact. It produced zero accepted and one retained rejected diagnostic record. See native_quality_witness.json. This finding concerns this source copy; it does not establish a defect in the published GFC algorithm generally. Historical CUDA.jl only calculated sizes with a different predictor and wrote zero payload, so its PASS was insufficient.

## Explicit corrected-reference secondary

Original source remains unchanged and blocked as the primary reference. native_corrected/kernels.h initializes each subchunk sign/count code and adds warp-shared read/synchronize/write/synchronize and phase-boundary barriers. The predictor, nibble wire format, byte-count2→3 convention, mapping and buffer sizes remain intact. native_codec_correction.patch and native_correction_provenance.json bind original, corrected, patch and actual witness hashes. Both static defects were repaired together; their individual causal contributions were not experimentally isolated.

## Certified contract

N512, eight64-value chunks, four blocks×two warps, dimensions1/2/3/32, source sequential and adversarial opaque-bit fixtures, R3. The original64million-value/255block×8warp/R100 CLI is unmeasured. Each of the four implementations has five checked in-process trials, explicit warmup, and mean trials2–5. The ROI sums eight compression-loop means with whole-device completion after every repeat, matching native CheckTest. Allocation, transfer, buffer reset and checking are outside. Component times were not serialized.

saved_output_audit.json checks all20 artifacts, each40,384UInt64 exports:4,224 exact actual-input/parameter prefix values,34,880 complete payload/offset values, and1,280 guards. All values match exactly. Every captured component is independently decoded (160 bit-exact round trips), and a source-format file is reconstructed and hash-checked (160 packet checks). These are reconstructed packets, not an assertion that the original CLI emitted files. GPU decompression was not ported or claimed.

## Mapping and performance boundaries

CUDA.jl uses its own shared-memory/warp APIs (framework_device_kernel). KA local-memory/index macros and JACC callbacks call CUDA-specific warp synchronization; both are hybrid_device_intrinsics. The chosen interop is disclosed, without claiming a framework feature is unsupported. Source scratch allocation1,536Int32=6,144bytes is retained despite two active warps/block. Register use and occupancy are unmeasured. Integer packing is not a floating-point roofline workload; no bandwidth or abstraction-only causal explanation is asserted.

The corrected-secondary mean microseconds are CUDA83.195833, CUDA.jl100.187583, KA125.098583, JACC172.009167. They sum all eight component means and do not describe the original huge production regime.

## Effort, provenance and retained material

Each current Julia model has one authored device-body candidate, one external GPU process/JIT check and zero compiler/runtime/numerical feedback repairs. CUDA.jl proactively restores the historical surrogate, separately from target translation. Native builds comprise one original diagnostic and one corrected accepted build; the composite reference repair is a shared phase. Historical attempts/tokens remain unknown. Prospective usage includes context/cache/coordination and overlaps workflow; saved-source proxies are not exclusive billed costs.

generate_oracle.py and fixture_provenance.json reproduce nineteen essential CPU fixture/oracle binaries (two input, sixteen payload/offset, one cuts). reference_original and reference_history preserve prior source/diagnostic adapter. Actual GPU diagnostic/output artifacts remain local, referenced by hashes; they are not source fixtures. Final config:study_config.json; final result:20261018T072321879425Z_fpdc. Read the top-level canonical guides and worker checkpoint for resumption.
