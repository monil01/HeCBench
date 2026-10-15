The original adaptive Fehlberg solver and CPU final equation coupling are
preserved. Each equation evaluation uses an IRIS task with two blocks of 32
threads running the original ECC and CAM device functions. The device time
argument remains int, including the original float-to-int conversion.

All device storage uses DMem. Each evaluation updates the host bindings for
the current solver stage and flushes both derivative and communication arrays.
The unchanged solver then performs its original host-side final coupling and
NaN/Inf compensation. No RNG is used.

`make run` keeps the original interval 100, workload 1, input paths
../data/myocyte/{y,params}.txt, and output.txt format. Validation runs the
native HIP sources with the documented BASELINE_FIX.md communication repair,
in an isolated copy with those same inputs,
and compares every trajectory entry from separate output directories.
