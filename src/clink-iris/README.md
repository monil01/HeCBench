Original LSTM recurrence, files, repeat argument, batch/sequence defaults and
1e-3 absolute reference tolerance are preserved. Input files come from the
original clink DVC archive; they are not committed.
For bounded equivalent validation build both host and kernels with
EXTRA_CFLAGS="-DCLINK_BATCHES=256 -DDEBUG", retaining all 20000 sequence steps and both
weight sets. Native validation uses an isolated copy with the same batch count.
The shared GPU lock is held only during GPU calls; CPU reference checks run
after release. DEBUG preserves the original first-sequence text dumps.
