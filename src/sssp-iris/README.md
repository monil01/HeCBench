IRIS dmem port retaining the original weighted SSSP frontier kernel and CPU/GPU
switching algorithm. Defaults and CLI options are unchanged. Host mutations call
DMem.update; partial queue and distance/color writes use iris_rw. The requested
-d device is checked for HIP backend 3 and AMD GPU type 256.

The original verifier checks every node cost against the supplied reference.
Native siblings remain untouched. The default NYR dataset is absent locally.
Bounded validation supplies a deterministic graph and independent Dijkstra
reference through the original -f/-c options, exercises warmup and repeated runs,
and sets -l 0 to force subsequent GPU frontier iterations. A second case uses
-l 8 to exercise CPU/GPU switching. These do not change source defaults.

