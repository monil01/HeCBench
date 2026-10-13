IRIS dmem port of the CUDA scatter benchmark. Run arguments and defaults
remain `10000000 100`. All five operations on int32, int64, float and double
reuse the source atomic reducers. The source's B=K=1 contiguous tensor
descriptor is flattened into a dmem index pointer without changing indexing.
Each repeat restores the reduction identity with DMem.update and uses rw.
Seed 123 is restored for each case after IRIS initialization.

The original one-sided PASS verdict is retained. FULL_CHECK additionally
compares every output element exactly with the same CPU reduction; source
values are +/-1, so all four types have exact arithmetic at validation sizes.
Local native validation uses an isolated copy with this same stronger check;
native siblings are unmodified. HIP runs require an IRIS HIP GPU device.
