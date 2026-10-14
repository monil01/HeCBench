IRIS dmem interval Newton solver. Both implementation choices and original
100000-repetition run defaults are retained. HIP interval arithmetic and
original Boost CPU reference checks are preserved; CPU code verifies results
and does not implement a device fallback. Bounded validation uses repeat=1.

Set HECBENCH_DUMP to write a uint64 equation count, all int32 result counts,
and each equation's defined lower/upper double endpoints. Both implementation
choices are verified against native HIP across every equation, excluding
unused output-buffer slots. Original CLI has no problem-size or seed option;
all 65536 equations and 1024 workgroups retain the original per-thread alpha.
