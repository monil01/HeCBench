# miniDGS: executable filtering is supported; EBADF recovery is unproven

## Recommendation

An ordinary, bounded NCU adaptation is source-supported: keep
`--target-processes all` and add an exact executable-name inclusion filter.
**Native must use `native_rank`, not `study_native`.** The three Julia versions
use `julia`. All four fresh plans are staged under
`/tmp/hecbench-roofline-v100-20261020/batch040-miniDGS`; the selector is
`process_filter_selector.json` there. No build, GPU run, test, source/MPI change,
transport change, write-guard relaxation or system modification was performed.

Installed NCU2022.4.1 help explicitly advertises `--target-processes-filter`,
comma-separated exact process names and regex expressions, and defines the
matched name as the executable name. The selected script launches
`mpiexec -n 2 rankcmd`. Native `rankcmd` is exactly
`STUDY_OUTPUT_DIR/native_rank canonical_fixtures`; Julia `rankcmd` is `julia`
with the same project and canonical main. Both rank executables match, while
Python, mpiexec and MPI utility executables do not match the profiling filter.
The filter changes profiling selection, **not** MPI rank creation or solver work.

The official adjacent-version MPI guide documents a single NCU around mpirun
for all ranks on one node ([NVIDIA CLI guide](https://docs.nvidia.com/nsight-compute/2023.1.1/pdf/NsightComputeCli.pdf)).
The exact proposed syntax is independently present in installed2022help; newer
`exclude:`/`exclude-tree:` syntax is not used. A fetch of the matching2022online
manual failed on its archive redirect and is not represented as evidence read.

## What is not established

The preserved023 assertion is `close_nointr(fd) != -EBADF` in `safe_close`;
the profiler process returned250. The log has no completed rank numerical rows
and no PID/rank attribution for the assertion. It does not establish GPU math
failure, an original source defect, or which launcher/rank process failed.

The installed filter help says which processes are *profiled*. It does not
promise that no tracking/injection machinery ever enters nonmatching parents.
Consequently this is not a claim that excluding launcher names fixes libsystemd,
PMIx/OpenMPI, descriptor handling or injection. The same failure may recur.
Original logs/counters/provenance are retained losslessly in this packet.

## Computational/replay invariants retained

The staged driver copies are byte-identical to the reviewed023 output-isolated
adapters; canonical sources, libraries, fixture hashes and helper bindings are
preserved. Only the fresh scratch prefix and NCU executable filter differ.
Both ranks still share GPU6, execute warmup and all five checked trials, all five
RK stages and four kernels/stage, MPI halo exchanges, barriers, input readbacks,
guards, full fields and existing independent oracle checks. No source-level serial loop replaces the two MPI ranks. NCU kernel replay and
profiling serialization remain measurement perturbations; MPI overlap or
ordinary simultaneous execution is not inferred. Instrumented times do not
replace the original unprofiled means.

The existing hash-bound conditional replay review is unchanged. Kernels are
finite and rank-local; host MPI sends/receives/waits occur between kernels and
use host buffers. All written device state must be restored for replay,
including Q/res/rhs/tmp/part and other relevant allocations. Filtering does not
weaken or independently prove that checkpoint requirement, tool/transport
compatibility or complete ROI coverage.

## Root-only future experiment

Wait for the original023 collector to finish and for foreign GPU6 activity to
clear. Preserve existing global GPU lock, UUID/NVML ownership and write guard.
Run the fresh plan root with the supplied selector and `--skip-qualified-selected`
so successful existing023 slots are not duplicated. `review.json` gives the exact
future CLI; it was **not executed** here.

The selector keeps first-mangled-function/geometry sampling and has
`expected_matching_launches: null`. Two GPU worker PIDs and both ranks' correct
trial1..5 outputs must be reviewed, but PID ordering cannot identify rank0/rank1.
Zero matched kernels with a normal host exit must not become a measured stage.
Any captured data remain partial first-invocation samples, not a source-counted
whole family or elapsed ROI. Preserve the original assertion and any new error.

A rank-local NCU strategy is not required just to use the documented single-node
all-rank mode. If filtering still fails, launching NCU inside MPI ranks would be
a **separate**, source-bound experiment with unique report names, the same two
ranks and full checks. It would still need the checkpoint/transport review and
could still fail; no such alternative is staged or declared successful here.
