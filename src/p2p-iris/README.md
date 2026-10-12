# p2p IRIS Port Status

Deferred.

The native HIP baseline builds on this machine, but `make -C src/p2p-hip run`
fails during peer-to-peer execution with repeated HIP `unspecified launch failure`
messages and prints `FAIL`. Since this benchmark's correctness depends on a
working multi-GPU peer-access HIP baseline, there is no passing native HIP
reference for validating an IRIS port in this environment.

No IRIS implementation is committed for this benchmark yet.
