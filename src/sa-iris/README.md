# Suffix array IRIS
Preserves dataset,size,repeat arguments, original1000000/100 run defaults,
DC3 recursion, integer sentinels, rank/merge kernels and checksum. IRIS DMem
replaces Thrust device_vector ownership. Stable bitonic sort-by-key and GPU
blocked exclusive scan replace Thrust sort/scan; sorting stability retained.
Partial-prefix writes use iris_rw to preserve zero sentinels. Host updates
before each repeat use DMem.update(host). SA_VALIDATION_DUMP optionally writes
all n suffix indices; validation checks exact native output and independent
lexicographic order. Bitonic/task overhead is a P1 performance compromise;
full default-scale workload is not claimed verified.
