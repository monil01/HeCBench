CUDA SimpleMOC host initialization, CLI and attenuate kernel are retained.
IRIS initializes before restoring source srand(2). Seven buffers use dmem;
flux/state and partially written scratch use rw. Launch block count preserves
the source's oversized expression and bounds guard. No default concurrency
or algorithm changes: source threads share state_flux and scratch, so default
multi-segment runs are racy as in native CUDA/HIP.

Deterministic validation uses the existing VERIFY=yes single-segment,
single-repeat path and -t 1 to initialize the source's nthreads input.
HECBENCH_ARTIFACT writes the complete state flux and fine flux arrays as
float32 for full native comparison. Native validation uses an isolated copy
with only this artifact dump added. Defaults and native siblings are unchanged.
