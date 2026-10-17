# Mean-shift reference audit

Historical CUDA.jl was a one-line alias of the unrelated mcpr touch/PASS stub. The current CUDA.jl implements the source global and tiled weighted Gaussian updates, retaining N=10000, D=3, squared-distance cutoff60, sigma4, 64 lanes,157 blocks and1000 synchronized updates. All10000 neighbors are traversed per valid point; the GPU does not exploit repeated rows.

The source CSV contains21 unique Float32 rows in3 widely separated clusters. The independent CPU oracle groups equal rows with exact multiplicities and computes the same mathematical recurrence in Float64 for1000 iterations. This is a CPU checking optimization only, not a replacement GPU algorithm. Every final position is compared, not only the source centroid PASS. The original centroid thresholds and known cluster membership supplement full30000-value CPU/dual evidence. The final1e-2 absolute position tolerance is predefined and much stricter than the original squared centroid distance10 threshold. Original Makefile --use_fast_math is not enabled in the named study compiler policy; original fast-math production performance is unmeasured.

## Independently justified native host-allocation repair (tiled only)

Original allocation has10000*3Float32 entries, but tiled kernel launches157*64=10048 lanes and reads data[3*tid+j] inside the tiled loops before the final tid<N output guard. Its final48 lanes therefore read beyond the original allocation. This is a static access witness from unchanged source, not an observed sanitizer or runtime failure. Barriers cannot be bypassed by returning those lanes early.

The named study allocates10048*3 entries and explicitly zero-initializes the144 padded values in both ping-pong buffers. This defines inactive-lane reads and preserves every valid query/neighbor computation and all source barriers; original kernel bodies and N remain unchanged. For tiled configuration this is a native reference host-allocation repair, native_reference_repaired=true, excluded from untouched-native aggregates. Original exact-allocation tiled CLI is not certified. Base kernel already guards tid<N; extra capacity is unused there, documented as named fixture storage capacity rather than algorithm repair. The actual padded input is serialized byte-exactly as30144 Float64 prefix values converted from the same Float32 host buffer copied to device.

## Diagnostics and effort

The first explicit10-update diagnostic retained all traces: base1.523ms/update, tiled1.179ms/update, CUDAstatus0. It justified keeping the full1000update primary contract. Diagnostic times are not primary series or selected fastest results. A metadata-edit Python SyntaxError was separately recorded; the shell launched the existing declared padded-base fixture, whose full series is preserved. No numerical tolerance or measured source was changed to favor a result. Reference restoration, adapters/oracle, native build/instrumentation and target translation phases are separate.
