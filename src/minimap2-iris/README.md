# minimap2 IRIS dmem port

Preserves the CUDA scheduler, packed anchor/control/return layouts, tiled kernel,
input/output arguments, USE_HOST_KERNEL option, and default input path.
Each batch is an IRIS HIP task; a 64-bit batch scalar replaces native pointer
offsets. Tracker DMem persists across tasks and output is read/write because each
task updates only one slice. Only HIP AMD GPU devices are selected.

Build with `make`; run with `IRIS_ARCHS=HIP make run`, or supply input and output
paths with `RUN_ARGS="input.txt output.txt"`. The repository's default
`../minimap2-sycl/in-1k.txt` is absent locally. Validation uses a documented
deterministic fixture in the local porting state, with varied lengths, tags,
coordinates, and widths including multiple tiles. Native HIP sources and
defaults are unchanged; the complete output artifact is compared exactly.
