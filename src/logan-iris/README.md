IRIS dmem port retaining original x-drop seed/extend kernel and CPU batching,
OpenMP input preprocessing, scoring, seed ABI, adaptive block sizes and CLI.
Original left/right extensions use serialized IRIS tasks per selected device
instead of overlapping native streams (performance compromise); multiple devices
and per-device preprocessing remain supported. All inputs, outputs and
antidiagonal scratch use dmem. Shared scratch capacity1024 covers original
maximum adaptive block size; source kernel algorithm is retained.

Validation supplies controlled seeded DNA pairs through original input CLI,
compares every score against unchanged native HIP, and runs one and two MI100
GPUs with OMP_NUM_THREADS1 for deterministic input order. Default kmer17/xdrop100
and ADAPTABLE behavior retained. Original low-xdrop<=31 kernel assumes64 entries
in warpReduce despite32-thread launch and is not used as a successful baseline.
Large100k input is absent; example input snapshot retained. CUDA unvalidated.

