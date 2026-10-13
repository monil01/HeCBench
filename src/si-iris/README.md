SI is deferred under the current restriction on opaque vendor-operation
replacements. The GPU set-intersection computation builds binary matrices on
the host and calls cublasSgemm (hipblasSgemm in the native HIP sibling). There
is no benchmark-owned device kernel to retain as an IRIS task.

No supplied IRIS dmem SGEMM library binding is present. Introducing a new GEMM
or intersection kernel would be an equivalent-algorithm replacement outside
the current authorization. Direct vendor calls from a native host allocation/
launch path would violate the dmem-only rule. No executable IRIS port or pass
is claimed.

Original dataset generation, ascending/descending datasets, partition default
10000, argument/output behavior, and all native/CUDA files are untouched.
The native CMake build also requires externally fetched cxxopts, fmt, Zipf
and progress-bar sources; they are absent from the checked local build caches.
An isolated offline configure assessment records that secondary prerequisite.
