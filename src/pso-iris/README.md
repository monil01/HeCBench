# pso IRIS Port

The IRIS host keeps the CUDA/HIP benchmark initialization and host reference
check. The particle update and personal-best update are launched as IRIS HIP
kernels over dmem buffers for positions, velocities, personal bests, and global
best.
