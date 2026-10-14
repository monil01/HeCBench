Original native paged-attention device templates, packed key layout,
FP32/BF16 conversion and xorshift/Box-Muller GPU initialization are retained.
The earlier simplified kernel is preserved in attempt03 history.
IRIS uses static LDS in place of dynamic extern shared memory, with
ATTENTION_MAX_SEQ_LEN8192 configurable at build time. Original default
4096sequence and all nine head-size choices are supported. The static LDS
adapter changes occupancy; native HIP kernels remain unchanged.
All buffers use DMem; input/output flushes support original CPU references
outside the GPU lock. ATTENTION_DUMP_DIR exports full output arrays.
All original make run arguments remain. Largest131072cache configuration
requires64GiB for the FP32 KV pair and cannot fit this32GiB GPU; bounded
equivalent native/IRIS validation preserves all program defaults.
