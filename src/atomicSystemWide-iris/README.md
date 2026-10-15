# atomicSystemWide: real concurrent CPU/GPU system atomics

The complete original64×256 GPU producers and16384 CPU producers operate on
the same ten counters. GPU work is an actual IRIS HIP task; storage, including
optional proof flags, is DMem-owned. CPU work retains the original `__sync`
operations. There is no CPU fallback, device-only replacement counter, native
allocation/copy/launch in the IRIS host driver, or sequential substitute.

## Build and run

Use a private IRIS checkout at `ff57e620a92fcf62eff3568d9c6eeaaf078f0340` with
patches0001–0004 and0006 applied, following their documents in
`tools/iris_runtime_patches`. Patch05 is not needed by this benchmark and was
not included in this worker's build. Do not overwrite a shared/live runtime.

```sh
make IRIS_ROOT=/path/to/private/patched/iris
IRIS_ARCHS=HIP IRIS_ASYNC=1 IRIS_NSTREAMS=4 HSA_FORCE_FINE_GRAIN_PCIE=1 ./main 1
make runtime_test IRIS_ROOT=/path/to/private/patched/iris
IRIS_ARCHS=HIP IRIS_ASYNC=1 IRIS_NSTREAMS=4 HSA_FORCE_FINE_GRAIN_PCIE=1 ./runtime_test
```

Wrap actual GPU executions in `flock -w 600 /tmp/hecbench-iris-validation.lock`
and bound each executable with `timeout 180s`. `HECBENCH_ATOMIC_PROOF=1` enables
a separate DMem progress buffer, GPU/CPU start handshake, per-thread GPU
completion count and observation of the active CPU interval. Both modes use
asynchronous task submission, original CPU work, then an actual task wait.
The CLI remains one positive loop-count argument; `make run` retains loop1000.
Counts whose original signed integer sum would overflow are rejected.

`verify_outputs.py --loops 10 --proof native.log iris.log` checks every counter
against a separate source-derived integer oracle. The original Exch/CAS results
are schedule-dependent and checked against their original legal ranges, not
against one particular native schedule. Disabled Inc/Dec remain disabled and
their two counters must remain zero. All other eight counters match exactly.

## Necessary transparent native repairs / P1 adaptations

The original native HIP source and old IRIS candidates/attempts are preserved
under `.porting-state/porting_logs/atomicSystemWide-iris`. Minimal repairs in
`../atomicSystemWide-hip/main.cu` are:

* Use coherent mapped pinned host memory rather than the conditional ordinary
  malloc/managed allocation; free with the matching `hipHostFree`.
* Replace only the unsupported GPU host-memory AND/OR/XOR instructions with
  equivalent real SYSTEM-scope atomic CAS loops. All per-thread operations,
  values and loop counts are retained; no reduction or operation aggregation.
* Define the intended32-bit OR mask as unsigned shift by `tid &31`, on both
  processors and in the verifier, instead of undefined oversized signed shifts.
* Use `__sync` atomic reads in the CPU min/max CAS loops; check every HIP API,
  launch and synchronization status. No ignored asynchronous failure.
* Optional proof hooks do not change any original counter operation. Ten raw
  counters are printed for independent validation; invalid/overflowing counts
  fail before launch.

Fine-grained VRAM was tested first with `hipExtMallocWithFlags` and
`HSA_FORCE_FINE_GRAIN_PCIE=1`. Allocation and CPU access succeed, but concurrent
CPU/GPU additions lose updates on **both** MI100s, despite exact independent
producer counts. GPU-only, CPU-only and serialized probes pass; uncached VRAM
has the same cross-agent failure. This is not an absent-GPU/absent-BAR blocker.
No unproven fine-grained-VRAM/kind3 runtime policy was introduced.

Coherent host memory passes simultaneous additions. Direct GPU bitwise host
instructions fail with a checked HIP launch error; their genuine system-CAS
equivalents pass per-operation and full-benchmark tests. See AMD's
[hardware/location/scope discussion](https://rocm.docs.amd.com/en/docs-7.2.1/reference/gpu-atomics-operation.html)
and, more importantly, the preserved concrete platform probes. The installed
HIP is7.0.1; documentation version7.2.1 is explicitly separate.

The original allocator was already conditional, and the benchmark contains no
HMM/page-fault measurement. Shared-counter/concurrency semantics are preserved,
but storage/migration behavior and instruction cost change (P1). PCIe CAS
contention is slow and variable: validated ten-loop native/IRIS runs take about
90/82seconds with proof enabled, not a performance equivalence or speedup
claim. Prior45/60-second CAS timeouts did not establish unsupported execution.

## Validation scope

AMD Instinct MI100/gfx908, real HIP backend. Complete loop1 and loop10 native
and IRIS runs pass all ten independent checks,16384 GPU thread completions,
16384×loops CPU/GPU updates each, overlap proof, and zero IRIS runtime errors.
The ownership regression checks80 GPU-generated counters, eight matching
DMem-owned frees, nontrivial GPU minimum, coherent pointer attributes and
invalid/freshness/policy/USM/region/dimensionality controls. Diagnostic sources
`finegrain_probe.hip.cpp` and `system_ops_probe.hip.cpp` are native platform
probes only, not benchmark completion paths.

Production loop1000 was not fully executed; no production-default performance
claim. The CUDA companion has matching kernel ABI, but CUDA compilation/run
was unavailable and is not claimed. Original CUDA baseline source is untouched.
