IRIS dmem grep retaining original Thompson-NFA parser, simplification, original
shared-memory device NFA construction and matching, limits, CLI/options and
runtests.sh default suite. State sizeof40/u32 sizeof4 asserted across backends;
match state contains only source null host pointers, and kernel builds device
pointers in original shared arrays. Result flags fully written on GPU, explicitly
flush_out to host; no CPU matching fallback. Backend proof goes to stderr to
preserve exact source matched-line stdout. Actual HIP backend3/type256 AMD MI100
required. CPU host visualizer/parser preserved for original CLI helper options.

Original small DVC testcase archive MD5/size and safe entries verified/extracted
in exclusive state, never writing shared native inputs. Validation executes all
original default patterns/files against unchanged native HIP and independent
GNU grep -E -x, comparing complete matched-line bytes. Duplicate invocations of
the default shell diffs may be collapsed by external validator, with identical
semantic cases retained. Bounded deterministic CLI cases supplement originals.
IRIS initialization is before source stopwatch (timing compromise); default
launch512x160 and all source algorithm/default behavior retained. CUDA backend
retained but NVIDIA unvalidated.
