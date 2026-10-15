Status: verified, HIP-only IRIS DMem with complete numerical output checks.

Pitch retains all11shape defaults and1000repetitions. HECBENCH_DUMP optionally
writes every logical output for pitched/simple2D/3D; padding is excluded.
HECBENCH_PITCH_SHAPES optionally bounds validation to the first1..11original
shapes. Explicit128-byte padded row strides are represented by IRIS dmem.
Original native CUDA/HIP allocations were read uninitialized, propagating NaNs.
Both native baselines now zero-initialize their allocations before the warmup,
matching the existing IRIS zero inputs. Initialization is outside kernel timers;
no sigmoid, indexing, launch geometry, CLI or default iteration count changed.
Untouched native sources and old failed attempts remain in local evidence.

All11 original shapes at repeat1 (plus original1warmup) pass full native/IRIS
bit-pattern comparison and independent CPU repeated-sigmoid reference checks:
252189530 logical floats in66 outputs, all finite, max CPU absolute error0.
Original1000repetitions on the first2 shapes also pass all1170650 floats in12
outputs with max error0. The complete11shape/1000repeat combination was not run.
Native dumps now use `HECBENCH_DUMP=prefix` and honor the same optional shape
limit. CPU artifact comparisons run after releasing each native/IRIS GPU lock.

Portability/performance distinction: IRIS represents pitched addressing with
explicit128-byte padded row strides in DMem, not native hipMallocPitch/3D
allocation APIs. Logical2D/3D output semantics are preserved, but physical
allocation policies, exact native pitches and allocator timings are not claimed
equivalent. IRIS task overhead is included in timing. CUDA source is retained
but NVIDIA execution was unavailable.
