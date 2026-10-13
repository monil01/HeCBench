IRIS DMem Langford search port. Preserves all default searches n7,8,11,12,15,
16383logical threads, four threads/block,100million result cap,64-bit counts,
compact layouts, CPU sorting/deduplication and published-result checks.

The five template instantiations have exported C kernel symbols for IRIS.
Counter updates and result retrieval use DMem update/flush; full result
buffer remains rw. Finalization precedes static teardown. Both native HIP
and IRIS match all five default counts:0,4,16,40,194. Initial unconverted
template-launch build failure and repair are retained in local attempt history.
