# CFD Euler solver with IRIS DMem

Retains original Euler initialization, step factor, flux and Runge-Kutta kernels,
mesh loader,2000defaultiterations,192thread grids,and density/momentum/energy
artifacts. Device-to-device old-state copy uses Task.dmem2dmem. Float3 remains
three32bitfloats; initialize_buffer val is explicitlyfloat. Bounds guard on
initialization handles rounded grid safely. Memory uses DMem exclusively.
Original DVC mesh97046cells recovered and verified in owned local state, no
shared dataset paths changed. Source padding-normal assignment bug is retained;
only valid mesh cells are compared/dumped, padded cells not reported by source.
Optional HECBENCH_CFD_ITERATIONS bounds iterationcount(default2000 unchanged),
HECBENCH_CFD_OUTPUT dumps all5fields for every valid cell. Isolated native
adapter adds only same iteration/dump hooks and missing original util.h.
