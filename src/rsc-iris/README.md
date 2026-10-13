IRIS dmem port retaining original CPU first-order-flow model fitting and GPU
RANSAC model evaluation, CLI/defaults, seed123, thresholds and candidate selection.
Each host reset and CPU-fitted model update calls DMem.update; partially written
candidate arrays and global count use rw. Shared outlier counter is equivalent
fixedone-int storage to the original dynamic launch.

RSC_DUMP optionally records the complete candidate(model,outliers) set and all
CPU model coefficients. Native validation adds only the same artifact hook;
native siblings remain unchanged. Bounded deterministic optical-flow input is
supplied via original-f option, with model-consistent vectors and controlled
outliers. Candidate order and tied best-model index vary legitimately with GPU
atomic scheduling; verification compares all candidate pairs sorted by model,
all coefficients, original CPU reference verdicts and best outlier count.
CUDA backend source retained but unvalidated on NVIDIA.

