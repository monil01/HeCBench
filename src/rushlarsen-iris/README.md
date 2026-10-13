Rush Larsen IRIS dmem port
==========================

Preserves the complete cardiac ODE integrator, source initial conditions, 19 double state variables per node, parameters, t=0, dt=0.02E-3, argument semantics, default 1000000 timesteps/1 node, Makefile run arguments 100/100000, and CPU reference. Each timestep submits the unchanged device integrator via IRIS. All states are rw; parameters r; time/dt are double and node count int. Full-state finite tolerance check uses atol=rtol=1e-9 against the CPU reference; source RMSE output retained. Optional HECBENCH_DUMP captures every GPU state double for native comparison.

Validation uses the existing CLI with 100 timesteps and 1025 nodes (a partial final block), comparing every state against an isolated native source adapter. The adapter only adds identical optional binary capture; no numerical/default changes and no native sibling edits. Full native/IRIS comparison uses atol=rtol=1e-10. IRIS_ARCHS=HIP and printed backend3/type256 AMD GPU proof are required.
