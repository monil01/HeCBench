LSQT is deferred because its default simulation inputs are unavailable.

The CUDA Makefile runs examples/input.txt, which lists simulation directories
containing model and run parameters. Neither examples/ nor examples.tar.gz
is present. Only examples.tar.gz.dvc is supplied (MD5
c7d1af7bc30a4faf929c8b399c071024, 35364096 bytes).

Without the real driver and model inputs, the native HIP baseline cannot
produce DOS/VAC/MSD/spin/LDOS artifacts for numerical comparison. Inventing a
replacement dataset would not validate the original workload. No executable
IRIS port or final_correct snapshot is claimed. Original CUDA/HIP files,
double precision, and the Makefile's DEBUG deterministic RNG setting remain
unchanged. This is a missing-input deferral, not an infrastructure failure.
