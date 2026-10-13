IRIS DMem N-body retains defaults16000 particles,10 steps,dt0.1,sample1,
MT19937 seed42 for position/velocity/mass, argument behavior and min3 steps.
Particle ABI10 floats=40bytes shared by host/HIP. Three original kernels are
ordered IRIS tasks per step, p/e preserved iris_rw; initial async input copies
become synchronous DMem updates. Energy explicitly flushed before host read.
Original CPU reference and energy absolute tolerance1e-3 retained. CUDA unvalidated;
reported timing includes IRIS submission and output flush overhead.
