# streamCreateCopyDestroy IRIS port

The prior invalid update-only timing candidate is preserved in
attempt02/previous_candidate. This implementation performs physical transfers
and preserves the original stream-lifetime graph.

Requires runtime patches0001..0004 including04's tested borrowed legacy-default
slot control (follow-upd2218958f). Every allocation is IRIS DMem. Baseline
uses the true HIP legacy null/default stream; explicit mode creates default-flag
physical streams, performs actual bound IRIS H2D tasks, synchronizes and destroys
them within each timed iteration. Initial runtime slots are removed outside
timing, not recreated before each measured copy. Runtime owns all stream handles.

Each requested copy marks its DMem host data valid/device dirty before H2D.
Backend physical counters assert the complete byte volume and exact successful
create/destroy counts. Unique physical buffers are allocated outside the timed
loop as in native HIP. Untimed IRIS HIP kernels read every copied device buffer
into a separate output DMem and flush it; the verifier asserts zero extra H2D
copies, so it cannot fix missed benchmark transfers by reuploading inputs.

Original defaults remain4096floats,256 base iterations,1/2/4/8 streams and
1/100/1000/5000 buffers, with17baseline plus17stream rows including warmups.
Optional HECBENCH_SCCD_VALUES,HECBENCH_SCCD_ITER_CAP andHECBENCH_SCCD_BUFFER_CAP
bound validation identically in native CUDA/HIP. HECBENCH_SCCD_DUMP writes every
validated buffer for all34rows; native hooks add untimed whole-buffer checks.

Performance qualification: H2D tasks wait before the next host-valid refresh
to prevent cache skips;128-command chunks support the full5000-buffer default.
IRIS scheduling/synchronization overhead serializes copy submission more than
literal native enqueues; stream overlap and timing equivalence are not claimed.
Stream-flag queries and physical-count instrumentation also contribute overhead;
these runs are correctness evidence, not native-performance parity claims.
The actual create/copy/sync/destroy operations, data volume and buffer semantics
are preserved. No native host allocation/copy/kernel launch substitutes for IRIS.

Validation covers all34rows at259floats/cap5buffers/cap2iterations and
4099floats/cap9buffers/cap3iterations; all34 complete artifacts per model match
native HIP and an independent CPU index-pattern oracle. Physical copies and
successful stream create/destroy counts match the original loops exactly in
each row. Runtime-default regression checks263 GPU-read values and invalid/
occupied controls, including unbinding without destroying the default stream.
The full uncapped5000-buffer production timing matrix is not claimed tested.
NVIDIA execution was unavailable; validation uses AMD MI100 HIP-only IRIS.
