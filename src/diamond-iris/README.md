IRIS DMem DIAMOND masking port. The full original CLI and CPU application
framework are built from immutable ../diamond-sycl/src sources, with all port
objects emitted locally under obj/. No shared framework source/object is changed.
main.cpp supplies the Masking bridge; the application entry remains the original
framework src/run/main.cpp. CUDA maskSequences, fixed33-residue sequence layout,
scalar parameters, original matrix-prefix transfer and CPU comparison behavior
are preserved. A persistent IRIS session requires an actual HIP GPU, DMem owns
all device buffers, sequences use iris_rw and are flushed before verification.
Host staging allocations are page-aligned/padded to isolate IRIS pinned ranges;
logical transfer lengths and contents remain those of CUDA/native HIP.
HECBENCH_DUMP optionally saves every device masking byte buffer for native
artifact comparison. The original Success check uses strncmp; full binary
comparison additionally checks all bytes against native HIP.
The provided ARDB database and short101-base reads support bounded cases;
long.fastq.gz needed by the unchanged default run is absent. Validation uses
8/16-read subsets and preserves the actual blastx arguments/output format.
Native HIP builds occur in benchmark-local copies of the full framework, so
its original Makefile cannot overwrite shared objects. AGPL notices retained.
