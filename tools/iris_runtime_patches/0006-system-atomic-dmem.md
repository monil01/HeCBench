# Patch06: DMem-owned coherent HIP CPU/GPU atomic storage

Reference: ORNL/iris `ff57e620a92fcf62eff3568d9c6eeaaf078f0340`, with committed
patches0001–0004 applied in order. Tested private root:
`/tmp/hecbench-iris-system-atomic-yDBNv9/work`. Neither `/tmp/iris`, the live
coordinator runtime, patch05, nor existing patch files were modified.

Apply in a separate source/build tree after04:

```sh
git apply --unidiff-zero --check /path/to/0006-system-atomic-dmem.patch
git apply --unidiff-zero /path/to/0006-system-atomic-dmem.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON=OFF \
  -DUSE_SIGHANDLER=OFF -DIRIS_ASYNC_STREAMING=ON
cmake --build build -j8
```

Patch04 itself needs the `patch -l -p1` application command documented in its
own markdown. Patch06 contains no05 changes. The coordinator may need to rebase
DeviceHIP.cpp hunk locations when applying after05; preserve both changes.

## API and ownership

`iris_dmem_bind_hip_shared_view(mem, device, &pointer)` accepts a **fresh contiguous
one-dimensional DMem root** (no tiles) with null host backing, allocation kind2, no enabled regions, no existing
physical allocation, and no USM flags. The selected device must be HIP. It
synchronizes submitted work, materializes the coherent pinned physical
allocation through DMem, and binds the host view to that same address. The host
view is borrowed: never free it, change host binding, enable USM, clear the
object, or access another device while borrowing. Calls are quiescent and must
be serialized against new submissions/host destruction. Initialize through the
view before submitting the first task; use atomic operations during overlap;
wait for tasks before releasing DMem.

DMem remains physical owner. No USM flag is set: USM would incorrectly suppress
the backend free. `host_ptr_owner_` stays false so ordinary `free()` cannot own
this HIP allocation. Existing kind2 `DeviceHIP::MemFree` calls `hipHostFree` in
the owning context. Device and host dirty state initially describes the same
storage; same-address kind2 H2D/D2H are no-ops, avoiding needless self-copies.
Ordinary distinct-address transfers, allocation kinds0/1 and default policies
are unchanged. This does not expose native application allocation/copy/launch
as a substitute for IRIS ownership or task execution.

## Why no kind3 fine-grained-VRAM mode

The requested fine-grained-VRAM possibility was tested **first**, not assumed
unsupported. `hipExtMallocWithFlags(hipDeviceMallocFinegrained)` succeeds and
its pointer is CPU-readable/writable with `HSA_FORCE_FINE_GRAIN_PCIE=1` here.
Nevertheless simultaneous CPU `__sync` and GPU `atomicAdd_system` lose updates.
Both MI100s reproduce it; all163840 GPU completions and all163840 CPU updates
are accounted for. CPU-only, GPU-only and serialized access pass. Uncached
VRAM gives the same concurrent failure. This is not missing BAR mapping or
missing GPU hardware, and successful fine-grained allocation is not proof of
cross-agent atomicity. A kind3 shared view was therefore **not** exposed as a
working system-atomic allocation policy.

Coherent host DRAM does support concurrent system-scope addition on this
platform. Per-operation tests show direct GPU bitwise AND/OR/XOR to that memory
return an actual HIP launch failure; other original operations and genuine
system-scope CAS equivalents complete. The full native benchmark passes after
the three bitwise operations are expressed as system-scope CAS retry loops.
The previous45/60-second timeouts were insufficient: the full ten-loop native
case takes about89seconds without proof instrumentation. There is no CPU
fallback, device-only counter, serialized CPU/GPU substitution, or ignored HIP
synchronization error.

AMD's hardware/location/scope explanation is at
<https://rocm.docs.amd.com/en/docs-7.2.1/reference/gpu-atomics-operation.html>.
It documents MI100 fine-grained memory forwarding and PCIe-operation limits;
our concrete VRAM atomicity result comes from actual tests, not a claim that
that document guarantees CPU locked-RMW atomicity to VRAM on this machine.
Installed HIP is7.0.1; the cited hardware discussion is7.2.1.

## Validation and qualifications

Consumer: `src/atomicSystemWide-iris`. Build with `IRIS_ROOT` set to the private
patched runtime. `runtime_test` performs actual GPU operations on all ten
counters in eight DMem allocations, checks coherent-host pointer metadata,
rejects invalid/freshness/policy controls, and proves all eight releases remove
HIP registration (successful pointer queries report Unregistered/type0,
flags0 after release). It uses HIP only for metadata queries.

The full original64-block ×256-thread, CPU+GPU algorithm is validated at
loops1 and10. `HECBENCH_ATOMIC_PROOF=1` enables a separate DMem-owned progress
allocation and CPU/GPU handshake: all16384 GPU threads complete their loops,
CPU completes16384×loops updates, and GPU execution observes the CPU interval
active. IRIS submission is asynchronous followed by a real task wait. All ten
outputs are independently checked; Exch/CAS retain their original legal ranges,
and the two originally disabled counters must remain zero. No exact equality
between nondeterministic Exch/CAS across separate runs is claimed.

All GPU executions use `/tmp/hecbench-iris-validation.lock` with bounded
executables; CPU oracle work is outside the lock. Detailed preserved evidence
is under `.porting-state/porting_logs/atomicSystemWide-iris/attempts`, notably
10(VRAM/coherence matrix),11(per-operation tests),12(native1/10), and subsequent
DMem/native proof and ownership regression attempts.

P1: coherent pinned host DRAM replaces the original conditional malloc/managed
allocation, and three direct GPU bitwise instructions become atomic CAS loops.
The same CPU/GPU shared-counter algorithm and original checks are retained;
there was no HMM/page-fault measurement in the source to preserve. PCIe CAS
contention is slow and timing varies substantially. Proof instrumentation and
IRIS scheduling add overhead; no performance equivalence/speedup is claimed.
The CLI/default loop1000 workload is retained but not fully timed/validated.
CUDA companion source is retained with matching ABI; no CUDA build/run is
claimed on this AMD host.

Base patch SHA256s:

```
0001 0ff057cec07701d5f128601804d8d3e6ab1f8dad2fedb9b92f9afb69891d7cef
0002 8e1ed23958891fcb246e28758ecc4fee3e6bae63bc6f0b4cdab407ce7da4c7c3
0003 9f680a5044039237381e2e072e2b051132f251ac0ef52c2d87f617944d9276be
0004 23efaf1c0fe9847c0a50a19ff546c82c60bb90f79abd27192f29cf0f1eaab11f
```
