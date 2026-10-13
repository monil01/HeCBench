QT clustering IRIS dmem port
===========================

Preserves the CUDA option parser, default 4096 points, geometry, candidate selection, synthetic data, cluster artifacts (p, p_seeds, p.N), and single-node communication behavior. IRIS initialization reseeds libc RNG; srandom(1) restores the source implicit default before generating points. Device scalar counts are explicitly narrowed to int as in CUDA kernel calls, although host counts use unsigned long. DataBuffer owns host storage and iris::DMem only; opaque handles never represent native GPU pointers. Kernel writes are partial and use iris_rw. Host uploads and resets call update; downloads use flush_out. HIP AMD selection is required and printed. CUDA backend kernels are supplied but not locally validated.

Validation uses --Verbose --SaveOutput on the default dataset and compares all cluster artifacts and the complete cardinality/seed-index sequence against an isolated unchanged native HIP source snapshot. No native sibling edits.
