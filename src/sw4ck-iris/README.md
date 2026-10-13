The original five SW4CK stencil kernels run as ordered IRIS task commands,
with the same ranges, blocks, double precision, coefficient slices, two
datasets, initialization, and numerical norm checks. Array storage uses DMem.
The output is read/write because kernels touch only selected components and
regions. Each outer repeat resets it with DMem.update(), and the final result
is flushed before the norm calculation. No RNG is used.

Default run arguments retain sw4ck.in and repeat 100. Validation can use
repeat 1 explicitly on the original two full datasets. The unmodified native
HIP source is built in an isolated copy; no native semantic adapter is needed.
