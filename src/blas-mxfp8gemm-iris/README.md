# MXFP8 E4M3 block-scaled mathematical IRIS DMem port

Preserves six original shapes, repeat CLI/default100, warmup1, alpha1/beta0,
E4M3 uniform1 input bytes0x38, UE8M0 scale127 per32 elements, column-major
BF16 output and original full-array rtol1e-2 check. Default inputs have no RNG.

All GPU storage/movement uses DMem. HIP tasks decode bytes and block scales
into FP32 DMem matrices; an IRIS bridge task calls real hipBLAS SGEMM with
those GPU pointers and the IRIS-owned stream; a HIP task rounds to BF16.
No native benchmark allocation/copy/launch and no CPU fallback. This P1
software-expanded operation is NOT hardware MXFP8; memory, accumulation and
performance differ and no native FP8 TFLOPS are reported.

`HECBENCH_IRIS_SHAPE=64,96,128 make run RUN_ARGS=2` bounds validation only.
`HECBENCH_IRIS_PATTERNED=1 HECBENCH_IRIS_SHAPE=17,29,256 make run RUN_ARGS=3`
exercises finite signs/subnormals and varying per-block scales. Every output
is compared to independent double dot products, with BF16 exact-match counts.
The extra fixture excludes E4M3 NaN encodings0x7f/0xff. Software scale indexing
is logical row/block, not a claim about nonuniform vendor-swizzled layouts.

Native block-scaled MXFP8 is unavailable on MI100/gfx908. Its unchanged native
probe result is recorded separately, not a numerical PASS or native runtime.
The six production default shapes are retained but not claimed run locally.
