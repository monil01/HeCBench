Retains six original native HIP radix/scan kernels, four-bit passes, workgroup
sizes, seed95123, default4194304 keys and repeat1000. Six buffers use dmem;
reordering uses rw and the last pass flushes every key. Source's complete
scan hierarchy is unchanged. FULL_CHECK compares every sorted key with a
CPU std::sort reference, checking the multiset as well as order. Optional
HECBENCH_ARTIFACT dumps all uint32 keys; isolated native copy only adds that
dump. Validation compares all4194304 keys exactly over two full sort repeats.
