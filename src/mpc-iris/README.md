MPC compresses and decompresses double-value files with original arguments, file names, bit-level algorithm, TPB=1024 and HIP warp-size specialization. IRIS DMem owns every device buffer. dmem.hpp maintains host backing stores, updates host-modified inputs and resets explicitly, preserves device intermediates with iris_rw, and flushes outputs through IRIS tasks. No native HIP allocation, copy or launch is used. Read-only HIP device-property queries preserve occupancy-derived block count. Build with make; IRIS_ARCHS=HIP make run RUN_ARGS='file [dimension]'. Default trace data is external and absent locally; bounded generated-double roundtrips are used for verification. IRIS synchronous task timing includes submission overhead.

Documented baseline repairs:

* HIP wave64 provides16 warps per1024-thread block, not32. The original
  dimensional prefix scan omitted channels17..32 and read nonexistent warp
  rows out of scratch bounds. Native HIP and IRIS now distribute all channels
  across the actual warp count and mask nonexistent rows. CUDA32's original
  dimensional scan already satisfies those assumptions and is unchanged.
* All three hosts now allocate for complete64-value plane padding plus one
  terminal plane-reset word. A source-derived CPU capacity proof showed that
  a64-value high-entropy input needs67 words, versus the original66-word
  allocation. This changes capacity only, not the compressed format or data.
* Native CUDA/HIP source line endings were normalized from CRLF to LF alongside
  these edits; untouched originals and all old attempts remain in local state.

Validation: all32 supported dimensions with8192 generated scientific doubles
pass exact native/IRIS compressed-artifact comparison, GPU roundtrips and an
independent CPU bitmap/bit-transpose/dimensional decoder. Partial1093-value
inputs cover dimensions1,4,17,31,32;64-value inputs cover1,32. Final allocation
hardening is additionally tested with high-entropy64/65-value inputs and
representative full/partial dimension32 cases. The original external default
trace is not claimed as tested. No CUDA execution was available.

Reproduce whole-bitpattern checks with `python3 verify.py original.bin
compression.txt decompression.txt`. The format can include one unused terminal
padding-reset word; native/IRIS complete-file comparison still checks that word.
CPU comparisons execute outside the shared GPU validation lock.
