Retains CUDA host/reference and all three double precision stencil kernels.
Thirteen buffers use dmem. Fluxes use rw to preserve unwritten four-cell
boundaries; each repeat resets via update before the three ordered tasks.
Input RNG runs before IRIS initialization. Default D=308 and repeat=100 are
unchanged. VALIDATION_D=20 bounds local checks using identical native adapter
dimensions. Every value of all five fluxes is dumped with HECBENCH_ARTIFACT
and compared, including boundaries, alongside all original RMS checks.
