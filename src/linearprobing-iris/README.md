IRIS DMem linear probing retains KeyValue(u32,u32), Murmur3, atomic insert/update,
delete tombstones, atomic iteration and full std-map verifier, MT19937 seed123,
insert/delete batch arguments and default16 8. Capacity default64*1024*1024 and
num_pairs=capacity/2 retained. Optional HECBENCH_HASH_CAPACITY (power of two) allows
bounded native/IRIS validation through a local copy of the HIP header; original
baseline snapshots unchanged. Every batch upload uses DMem.update(host); hash table
remains iris_rw across batches, outputs flush before map verification.
Timing includes IRIS task overhead; CUDA backend source unvalidated.
