# p2p: IRIS-owned peer copies and true remote kernel reads

Build a private IRIS runtime at `ff57e620a92fcf62eff3568d9c6eeaaf078f0340`
with patches0001–0007 in `tools/iris_runtime_patches`, following their exact
application commands. Then:

```sh
make IRIS_ROOT=/path/to/patched/iris
HSA_FORCE_FINE_GRAIN_PCIE=1 IRIS_ARCHS=HIP ./main 2
make run IRIS_ROOT=/path/to/patched/iris # original repeat2000 default
```

Two peer-capable HIP GPUs are required. GPU storage is exclusively DMem-owned;
the driver submits IRIS kernels and calls IRIS peer-control APIs. Native HIP
is used only for read-only capability queries. The runtime performs explicit
peer copies and launches both original multiply-by-two kernels with the input
pointer still on the other GPU. Nonmaterializing allocation queries reject any
silent local replica. No CPU kernel fallback or native host GPU launch/copy.

The two64MB buffers and original CLI/default repeat count are retained.
`HECBENCH_P2P_ELEMENTS` is an optional compile-time bounded fixture size;
the unchanged default is16,777,216 floats per device. The partial-block guard
permits odd fixture sizes without changing default kernel arithmetic.

## Validation

Two MI100/gfx908 GPUs, ROCm7.0.1. Full-size repeat2 checks all16,777,216
copy-phase values and33,554,432 remote-kernel outputs. Both output arrays are
byte-exact against native HIP and independently checked against
`(index %4096)*4` and `*2`. An odd4099-element repeat3 case covers kernel tails
and odd copy counts. Invalid output/scalar/unallocated/post-submit remote
bindings are rejected. Every run checks IRIS errors and successful teardown.
Production repeat2000 is retained but not claimed as fully timed/validated.

`make runtime_test IRIS_ROOT=/path/to/patched/iris` builds an additional
two-kernel-in-one-task rejection regression. Remote-bound tasks explicitly
require one kernel: binding, task submission, and both graph submission routes
reject invalid compositions before enqueueing, including a kernel appended
after binding. Patch07 also resolves the active command rather than the task's
first command. This is an API-rejection test; the normal benchmark separately
checks all33,554,432 valid remote-kernel outputs and forbidden opposite-device
allocations. Generic multi-kernel remote execution is not supported or claimed.

Optional `HECBENCH_P2P_DUMP=/absolute/prefix` writes complete output artifacts.
Native CUDA/HIP sources include an opt-in observation helper activated with
`-DHECBENCH_P2P_VALIDATE`; default computation is unchanged. Run:

```sh
python3 verify.py /absolute/native-prefix /absolute/iris-prefix --elements 16777216
```

The verifier discovers all dumped pairs, requires matching native/IRIS pair
sets and both outputs per pair, and checks every finite output against native
and an independent source-pattern oracle, not a checksum or timing label.
The two-device host certifies pair0/1, not arbitrary GPU topologies. The harness
explicitly rejects HIP_DEVICE_FACTOR other than1 rather than treating logical
IRIS indices as physical HIP ordinals. CUDA source receives
the matching optional hook and bounds guard; no CUDA build/run is claimed.
Historical failed candidates and the original native sources remain in
`.porting-state/porting_logs/p2p-iris`.

## Explicit P1/environment qualifications

`HSA_FORCE_FINE_GRAIN_PCIE=1` must precede HIP initialization on this PCIe
system. It recovers the unchanged native reference as well as IRIS execution;
see AMD's [RCCL usage tips](https://rocmdocs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html).
The native copy-phase contents were initially uninitialized; IRIS defines a
pattern so that every copy is meaningful and independently validated. The
subsequent original mathematical input and both remote kernels are preserved.

IRIS uses explicit `hipMemcpyPeer` and supported device synchronization instead
of native `hipMemcpyDefault`. Control operations and remote kernels use a
quiescent synchronous contract. Copy routing and scheduling costs differ;
no bandwidth-equivalence or speedup claim. Peer-context synchronization with
`hipCtxSynchronize` returned801 on this installation; the supported
`hipDeviceSynchronize` implementation passes. All storage remains IRIS-owned.
