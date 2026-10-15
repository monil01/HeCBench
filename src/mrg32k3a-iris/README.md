# mrg32k3a: IRIS DMem + hipRAND

Preserves the complete warmup, timed host generation, timed device generation,
and full host/device comparison. CLI is `./main n repeat`; `make run` defaults
to `1000000 10`. Generator type is hipRAND MRG32K3A, ordering PSEUDO_BEST,
seed1234, offset0. As in the original, every call creates a fresh generator and
generates n successive values; repeats restart the same sequence. There is no
replacement PRNG, arbitrary seed change, initializer-only port, or GPU fallback.

Actual `hiprandGenerateUniform` runs inside the shared IRIS vendor bridge, on
the IRIS stream and a fresh DMem-owned output array. The result is flushed in
the same task. Vendor opaque internal generator state remains library-owned;
no caller GPU buffer uses native host allocations or copies. The original
hipRAND host generator path is retained as a timed workload and full-array
oracle, not used to fill the device result.
Each callback verifies the resolved pointer is physical device memory on its
current HIP device. Actual successful generation calls and values are counted
and required to equal `(repeat+1)` and `(repeat+1)*n`, respectively.

The native source overwrites/leaks all but its last generator handle. The port
destroys each fresh host/device generator after its final use, preserving all
generated values and restart/state progression semantics. Native numerical
behavior is otherwise untouched; its only changes are opt-in trace hooks.
IRIS scheduling, stream synchronization and per-call generator destruction add
timing overhead; no performance equivalence to the leaking native lifecycle
is claimed. With tracing enabled, both reported timings also include dump I/O.

Build: `make IRIS_ROOT=/path/to/iris`; run with `IRIS_ARCHS=HIP` in this directory.
Optional matched `HECBENCH_MRG_DUMP=/path/prefix` writes every value of every
warmup and measured generation to `.host.bin` and `.device.bin` files. Each
contains `(repeat+1)*n` float32 values in call order, allowing complete native
comparison and checking the restart behavior across repetitions.

Validation covers n=1,259,4099,65537,1000000 with repeat=1,2,3,5,10 respectively.
Every warmup/measured host/device value (22,820,794 floats) is byte-exact with
native HIP. Every device value also equals the original host generator oracle,
and every fresh-call trace restarts identically. The original default is also
run without tracing. Physical device-pointer checks and actual GPU generation
counts pass. AMD HIP is validated; CUDA vendor sequence equivalence is untested.
