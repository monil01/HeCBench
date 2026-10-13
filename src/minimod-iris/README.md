IRIS dmem preserves CUDA seven-region timestep, source injection, buffer
swap, CLI, coefficients and defaults. Five GPU grids use dmem; partial
updates of v/phi use rw. ldimx/y/z explicitly narrow to int as in the source
kernel signature; llint remains 64-bit. Host lead padding is retained,
while dmem binds the useful logical grid.

The source reads source[nsteps] beyond its nsteps allocation and copies
partially initialized phi/vp/eta padding. VALIDATION=yes explicitly zeroes
host padding and allocates one zero source padding slot. The isolated native
copy applies the same initialization only; algorithm, defaults and native
originals are unchanged. --finalio dumps the complete physical wavefield
for grid64/nsteps3 validation with warmup and two iterations. Uninitialized
default padding/source behavior remains a source limitation.
