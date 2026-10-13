# Particle filter IRIS DMem port

Preserves original likelihood, sum, normalize and find-index device kernels,
LCG seed/video initialization, CLI and output.txt. All arrays use DMem and
persistent read/write state across frames. -fwrapv retains intended signed LCG
wraparound in host code. IRIS initializes after original video generation.
HECBENCH_PARTICLES_OUTPUT optionally writes full arrayX,arrayY,weights arrays.
Native isolated adapter adds only this full-output hook and missing kernel
headers copied verbatim. Bounded deterministic validation uses 256 particles
(one block), 32x32 video, four frames; this avoids the original normalize kernel's
lack of inter-block synchronization without changing its algorithm or defaults.
Default multi-block execution retains that native synchronization limitation.
