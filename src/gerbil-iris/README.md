# IRIS Gerbil

Preserves the original complete two-phase k-mer counting pipeline, including
CPU input partitioning and its hybrid CPU/GPU scheduler. Persistent GPU buffers
are IRIS DMem. Original insertion/compression kernels dispatch via IRIS tasks
and its vendor-kernel API; overflow sorting uses real GPU Thrust on the IRIS
stream. Hidden DMem bindings track every embedded buffer pointer. No shared
runtime changes or CPU replacement for GPU counting.

P1: the first visible HIP GPU is used; task/copy completion is synchronous and
host callers are serialized across GPU submissions. Original native stream
objects are not used for IRIS computation. GPU byte gather/scatter/fill tasks
handle partial slab transfers through small DMem staging buffers, avoiding
host copies of unused capacity. The source's adaptive 80%-of-free-GPU-memory
reservation remains unchanged; numerical throughput is not performance-equivalent
to the original stream pipeline. GPU insertion launch traces must be present
for a valid GPU test; a CPU-only scheduler outcome is not accepted.

Build requires CPU Boost system/thread/regex/filesystem libraries and HIP/IRIS.
Set BOOST_LIBDIR/BOOST_LIBS for installed libraries. A benchmark-local extraction
of matching Ubuntu runtime packages can supply missing Boost dependencies;
no system installation is required. On the validated Ubuntu 24.04 x86-64 host,
`python3 fetch_boost.py` recovers the five checksum-pinned runtime packages into
`.deps/`; development headers must already be installed. These are CPU support
libraries, not a substitute for GPU execution. Original FASTQ input:
https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR164/ERR164407/ERR164407.fastq.gz
Original CLI/defaults are unchanged; override RUN_ARGS for a bounded original
read subset. Tests must compare all output k-mer counts, not timing alone.

Minimal baseline repair: the legacy 32-thread warp-synchronous insertion kernel
communicated through non-volatile shared-memory views. On this gfx908/compiler
the original native kernel returned zero keys (including failure to merge a
duplicate in a four-key diagnostic). Making its shared array and communication
pointers volatile restores the intended memory accesses, without changing the
hashing/probing/counting algorithm. The same five declarations are repaired in
the assigned CUDA/HIP baselines and this port; untouched snapshots and rejected
barrier-only diagnostic remain in local evidence. Numerical acceptance requires
whole-output independent counts, not merely agreement with that repaired native
baseline. This does not claim portability of legacy warp-synchronous code to
arbitrary independently scheduled architectures.
