# NAS IS with IRIS DMem

Retains class S default65536 keys, all24iterations, generator, seven rank kernels
and full sorted-key verification. CLI three workgroup sizes retained. Dynamic
scratch is statically bounded to2048ints per block, covering valid HIP block
sizes1..1024 without changing scan/reduction logic (P1 storage compromise).
All partially written/in-place arrays use DMem read/write; scalar generator
seed/multiplier remain double and rank widths int. HECBENCH_SORTED_OUTPUT
optionally dumps every sorted key. Isolated native adapter adds only this dump
and supplies missing kernel/constants headers from CUDA source verbatim.
