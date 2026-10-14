Original bit-factor kernels, KISS RNG, factor updates, CLI, stopping logic and
confusion calculation remain. Device templates are behind IRIS wrappers;
host allocation/copy/flush use registered DMem host buffers and synchronous
IRIS tasks. CPU confusion/artifact checks run outside the shared GPU lock.
HIP device properties are queried only for original optional load-balancing
metadata; HIP does not allocate, copy or launch the computation.
WARPSPERBLOCK8 matches the native HIP build. Make run retains original MNIST
arguments; validation uses original built-in test matrix5000x5000, seed123,
one run and two iterations with112 lines, preserving program input/defaults.
