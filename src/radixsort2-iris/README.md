# Radixsort2 IRIS
Preserves original option parser, float/uint and keys-only/pair modes, mt19937
seed19937, element default1048576, iteration defaults100/10, quiet/help and
keybits printing (original keybits does not restrict sort). Original run target
four536870912-element modes retained; RUN_ARGS selects a bounded invocation.
Stable32-bit LSD radix sort uses IRIS HIP flag, scan and scatter tasks with DMem.
Float input is original uniform[0,1), so IEEE positive bits preserve ordering.
Per-iteration host inputs refreshed with DMem.update(host). Original sorted-key
PASS check retained; RADIX_VALIDATION_DUMP optionally saves complete keys/pairs
for native HIP and independent stable-sort validation. GPU scan/task overhead
is a performance difference; full large defaults not claimed verified.
