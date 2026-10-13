IRIS dmem PRNA port retaining original partition-function kernels, thermodynamic
parameter parser, FLOAT representation, canonical pairing, matrix/ProbKnot output
and CLI/defaults. Native HIP kernel/reduction branches are retained under exported
wrappers. The w5[-1] sentinel layout is preserved: dmem stores the allocation base,
wrappers add1 before calling original kernel bodies. Scratch uses dmem with native
uninitialized allocation semantics; partial writes use rw, const inputs use r.
All original fill phases launch as HIP tasks on device0; host serial fill is not
used. No host vendor operations are substituted.

DATAPATH must reference original RNAstructure tables. DVC thermodynamic/test/HIV
archives were recovered, checked by MD5/size, inspected and extracted in exclusive
scratch without native source edits. Validation compares complete probability
matrices, log probabilities and ProbKnot structures with native HIP on bounded
original-sequence data, plus an attempted default HIV run with a bounded timeout.
Default input filenames/options are retained; supply assets through original CLI.
CUDA backend source retained but unvalidated on NVIDIA.

