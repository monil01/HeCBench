# Patch07: command-local remote bindings, including multi-kernel tasks

Apply after the exact committed01–06 source sequence with
`git apply --unidiff-zero --check <patch>` then
`git apply --unidiff-zero <patch>`; rebuild in an isolated source/build tree.
The reference stock revision remains ff57e620a92fcf62eff3568d9c6eeaaf078f0340.

Independent semantic review found that05's pointer resolution used
`task->cmd_kernel()`, the task's first kernel command, even when preparing a
later kernel. Transfer skipping already used the actual command, so a binding
could leak into an unrelated subsequent command. Original p2p uses one kernel
per task and its retained numerical evidence was unaffected; general use of
multi-command tasks was unsafe.

The supported API explicitly requires exactly one kernel in a remote-bound
task. Binding rejects an existing multi-kernel task; both TaskSubmit overloads
and both GraphSubmit overloads validate all tasks before enqueueing, including
recursive subtasks. This also rejects a second kernel appended after binding.
The null-graph immediate-submit route propagates TaskSubmit's error instead
of discarding it and falsely returning success.
Ordinary tasks without remote bindings retain their previous composition rules.

As defense in depth, this patch records the actual `Command*` at `DeviceHIP::KernelLaunchInit` and
uses that command's binding map for `KernelSetMem`. Each device worker serializes
argument setup/launch (as it already does for params_); no shared task-global
pointer map or kernel argument ABI change is introduced. The pointer is read
only during that command's setup and is replaced before the next setup. It is
not a retained allocation owner or asynchronous completion callback pointer.
Root remote inputs with nonzero argument offsets are now explicitly rejected,
matching the documented whole-root API contract.

`src/p2p-iris/runtime_test.cpp` binds a remote first kernel then appends a
second kernel. Binding again, direct submission, graph submission and ordered
graph submission must all reject this composition before enqueueing, with no
runtime errors or wrong-device allocation. This is an API-control test, not
numerical execution of the invalid composition. The separate full p2p benchmark
checks all33,554,432 valid remote-kernel outputs (2x/4x) against native and an
independent oracle. Generic multi-kernel remote execution is not certified.

Preserved attempt07 experiments show the original runtime and an initial
active-command-only variant both time out when executing that composition.
Source inspection also finds first-command-dependent output bookkeeping in
upstream Device.cpp. The final solution deliberately rejects the unsupported
composition before scheduling; it does not claim to fix every generic IRIS
multi-command scheduling/dataflow issue. No timeout is counted as a pass.
The upstream unsubmitted-graph `GraphRelease` callback also deadlocks in the
negative test; it uses the existing direct `iris_graph_free` and explicit task
release APIs instead. This cleanup adaptation is not a claimed generic graph
release fix. The task-list query returns the number of entries, not zero;
earlier diagnostic expectation errors are retained as failed runs.

Build/run, using the freshly patched runtime:

```sh
make -C src/p2p-iris IRIS_ROOT=/path/to/patched/iris runtime_test
cd src/p2p-iris
HSA_FORCE_FINE_GRAIN_PCIE=1 IRIS_ARCHS=HIP ./runtime_test
```

The main peer harness explicitly rejects HIP_DEVICE_FACTOR values other than1:
its source enumerates physical devices and does not promise virtual/logical
ordinal mappings. The external verifier now discovers every dumped pair,
requires exactly matching native/IRIS pair/output sets, and checks both arrays
for each pair against native and the independent pattern. More-than-two-device
GPU execution remains untested on this two-device host; parser enumeration
does not claim arbitrary-topology physical validation.
