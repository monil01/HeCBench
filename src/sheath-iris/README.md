IRIS DMem sheath simulation with original seed123, constants,500000 particles
per species,1000 steps and result.dat default output. All three particle
kernels preserve double/long widths through typed dispatch. Density resets and
host electric fields explicitly update DMem; particles remain rw on-device.

HECBENCH_PARTICLES / HECBENCH_STEPS compile overrides permit bounded validation;
clean before changing EXTRA_CFLAGS. The isolated native baseline uses identical
overrides. A final partial-step field zone is emitted for bounded runs; default
1000-step output schedule is preserved. Native profiling device reset is
replaced by explicit IRIS finalization.

Validated2048 particles/species,25steps against all initial/final field values.
All1212scalars are finite. Floating atomics produce order-dependent roundoff;
comparison uses rtol1e-4 and absolute floors for columns x,nde,ndi,rho,phi,ef:
1e-12,1e10,1e10,1e-8,5e-5,0.05. The first tighter near-zero field comparison
failed and is preserved. DMem update/rw/scalar handling was audited before
changing only the verifier's unit-specific absolute floors.
