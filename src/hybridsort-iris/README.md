Hybrid bucket/merge sorting using IRIS dmem
==========================================

Preserves source histogram/pivot/bucket/prefix/float4 merge/pack algorithm, workgroup sizes, argument handling, exact full-array qsort reference check, and default 50,000,000 elements. Device buffers are host storage plus iris::DMem. Partial writes use iris_rw; host copies use update and flush_out. Restores source implicit srand(1) after IRIS init. Source kernel float4 ABI is represented by an aligned 16-byte host struct; kernels use backend float4.

Bounded validation compiles with -DHECBENCH_SIZE=131072, supported via an optional preprocessor override. An isolated native source snapshot adapter changes only its SIZE constant to the same value and adds the same optional HECBENCH_DUMP full sorted-float binary capture. Default builds and native siblings are unchanged. Both runs sort identical random input, perform source exact comparison against qsort for every element, and compare every sorted float byte-for-byte. HECBENCH_DUMP is optional and does not alter computation.
