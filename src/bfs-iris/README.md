# BFS IRIS
Uses IRIS DMem and two ordered HIP tasks per frontier, retaining the CUDA
kernels, 256-thread launch geometry, graph format, forced source node 0,
ignored edge weights, and exact CPU reference distance comparison.
The convergence byte is updated from the host before each level and flushed
back after frontier promotion. Distances are flushed before verification.
The printed execution time includes IRIS task submission and convergence flush.

Build: `make`. Run: `IRIS_ARCHS=HIP make run`, or
`IRIS_ARCHS=HIP ./main graph.txt`. The original default input path is retained.
The installed default graph currently contains 4096 nodes.
Validation must use `flock /tmp/hecbench-iris-validation.lock` and a timeout.
No random initialization is used. CUDA backend source is provided but untested.
