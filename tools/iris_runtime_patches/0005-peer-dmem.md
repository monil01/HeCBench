# Patch05: actual peer copies and remote DMem kernel reads

Reference: ORNL/iris `ff57e620a92fcf62eff3568d9c6eeaaf078f0340`, after
patches0001–0004 in order, including the final borrowed-default-stream version
of04. Apply05 with `git apply --unidiff-zero --ignore-space-change --check <patch>`
then `git apply --unidiff-zero --ignore-space-change <patch>`, in a private
source/build tree. The whitespace flag accommodates one upstream removed line
with trailing whitespace without embedding that whitespace in this patch. This
zero-context patch is for the exact cumulative baseline, not arbitrary IRIS
versions. Rebuild the runtime before compiling `src/p2p-iris` against it.

## Ownership and synchronous control contract

* `iris_hip_enable_peer_access(device, peer)` enables actual HIP-context peer
  access after synchronizing submitted work; same/non-HIP/invalid devices fail.
* `iris_dmem_peer_copy(dst, dst_device, src, src_device, bytes)` operates on
  whole, equal-sized root DMem objects on different HIP devices. Source storage
  must already exist and be valid; destination storage is materialized by IRIS.
  IRIS performs `hipMemcpyPeer`, then supported `hipDeviceSynchronize`, and
  marks destination device data valid and its host data dirty. Applications do
  not allocate, copy, or launch through borrowed native data pointers.
* `iris_task_bind_hip_remote_dmem(task, argument, source_device)` binds an
  unsubmitted kernel's read-only whole root DMem argument to an already valid
  allocation on the source GPU. The command records this source-device binding,
  skips input migration, and resolves the original remote GPU pointer at launch.
  Output/scalar/region/dirty/unallocated arguments and submitted tasks fail.
* `iris_dmem_is_allocated_on_device` is a nonmaterializing observation used to
  prove that remote reads did not silently become local copies.

Controls are quiescent: complete prior producers and serialize calls against
new submissions, host access, and destruction. Remote-bound tasks are explicitly
synchronous; this is not a new asynchronous cross-device dependency protocol.
Ordinary tasks retain their previous transfer behavior. DMem owns all storage.
No kernel argument ABI change, raw host-side peer pointer, or CPU fallback.

## Actual validation and qualifications

Two AMD MI100/gfx908 devices, ROCm7.0.1, `IRIS_ARCHS=HIP` and
`HSA_FORCE_FINE_GRAIN_PCIE=1`. The latter must be set before HIP initialization;
without it the unchanged native benchmark fails on this PCIe system. See AMD's
[RCCL usage tips](https://rocmdocs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html).

The original64MB-per-device buffers pass in both directions:16,777,216 copy
values and33,554,432 remote-kernel outputs. Both output arrays are compared
in full with native HIP artifacts and an independent modulo4096 pattern
oracle. Nonmaterializing queries prove no source replica exists on the other
GPU. Negative API tests reject output/scalar/unallocated/post-submit bindings.
An odd4099-element fixture additionally covers the final partial workgroup.

P1: the backend uses explicit `hipMemcpyPeer` plus device synchronization,
whereas native uses `hipMemcpyDefault`. Copy timing/API routing differs;
no bandwidth-equivalence or speedup claim. Undefined initial native copy-phase
contents are replaced by a defined pattern to validate every copied value.
The subsequent two remote kernels and their original mathematical input are
unchanged. CLI repeat2000 and full-size defaults remain; validation fixtures
are explicitly bounded. Historical failed synchronizations are retained:
`hipCtxSynchronize` returned801 on this installation; the implementation uses
the supported device synchronization API instead.
