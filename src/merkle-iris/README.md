IRIS dmem merkle preserves original Rescue-Prime finite-field math, round constants,
vector layout, original phase0/phase1 indexing, seed19937, original 4rounds and
1M..16M-leaf default sweep. Original GPU helpers/kernels retained together in a
single backend compilation unit, without opaque library replacements. Scalar
size_t64/int32 widths and ulong64/ulong4 size32 ABI asserted; partial internal
node writes use rw. VERIFY=yes retains source DEBUG256-leaf complete output and
uses explicit dmem.flush_out. Debug output indices0..3 are original uncomputed
node0 allocation bytes, not a numeric result; verifier checks all computed
255nodes/1020field words exactly against unchanged native HIP. No CPU hashing
fallback. Actual HIP backend3/type256 AMD MI100 required. CUDA retained, NVIDIA
unvalidated. Default release sweep gets a bounded native/IRIS timed attempt;
no timeout is claimed as a successful complete default baseline.
