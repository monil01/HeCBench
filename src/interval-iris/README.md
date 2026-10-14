IRIS dmem interval Newton solver. Both implementation choices and original
100000-repetition run defaults are retained. HIP interval arithmetic and
original Boost CPU reference checks are preserved; CPU code verifies results
and does not implement a device fallback. Bounded validation uses repeat=1.
