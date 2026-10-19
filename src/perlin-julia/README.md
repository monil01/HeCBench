# Perlin noise: original-width multistream strips

This is the original gradient/fractal-noise algorithm with the original 61,440-pixel width, source 16×16 block mapping, and all six stream counts. The accepted contract uses 512-row strips, a 16-row integer-lattice control, and a 48-row seeded single-octave control. The original 34,560-row CLI separately ran successfully; it is not independently certified by the strip evidence.

`contract.json` predates target measurements. Full UInt8 RGBA fields are compared with the predeclared absolute tolerance of one grayscale level. Actual inputs, initial outputs, hash/gradient readbacks, guards, RGB equality, alpha 255, lattice gray 127, and within-model stream-count identity are checked exactly. A separate host-only Float64 mathematical oracle checks every pixel. This is quantized-field accuracy, not bit identity or validation of an unobserved floating-point field.

## Execution and evidence

Build the host-only oracle library using the exact command in `checker_build_recipe.json`, then run the corresponding `study_config_*.json` through `tools/verify_portable_julia.py`. These configs reference the study machine; regenerate absolute paths for another checkout while preserving source contents and predeclared contract. The compiled library and multi-gigabyte captures remain ignored; the source/recipe are retained. Readbacks and the oracle run outside the timed region. Five independently checked trials are retained; timings use only trials 2–5.

The measured region sums eight multistream launch-and-whole-context-completion regions. Buffers, stream creation, upload, parameter observation, readback and oracle checks are outside. CUDA.jl uses explicit CUDA streams; KA uses a CUDA host stream context around genuine KA kernels, and JACC uses its CUDA launch specification. These host backend boundaries do not establish cross-vendor portability. The GPU math is translated Julia, with no native GPU delegation; the C ABI is exclusively a CPU checker.

Native static readonly device hash/gradient symbols become explicit Julia readonly device arrays. UInt8 storage, Float32 arithmetic, source mixed Float64 normalization literals and truncation are recorded separately. There is no counter-derived roofline or causal abstraction-performance claim. Actual token windows include inherited/cached context; interleaved coordinator work is excluded from clean calibration.
