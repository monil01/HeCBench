# aop IRIS port status

Status: deferred.

The native HIP benchmark builds and runs successfully, but this benchmark is
not a direct IRIS dmem-only translation candidate in its current form. The
HIP/CUDA implementation launches templated kernels and passes pointer-sliced
device aliases such as `d_paths + timestep * num_paths`, `d_svds + 16 *
timestep`, and `d_cashflows = d_paths + (num_timesteps - 1) * num_paths`.

IRIS dmem task parameters represent whole data-memory objects, not arbitrary
native device pointer aliases. A faithful port would need a broader kernel ABI
rewrite that replaces those pointer slices with explicit base buffers and
offset scalar parameters for every templated kernel specialization. That is
larger than a mechanical dmem port and should be handled as a dedicated rewrite
with separate numeric validation.

Native HIP validation reference from this pass:

```text
./main -runs 100
GPU Longstaff-Schwartz: 0.44747600
Binonmial             : 0.44880498
European Price        : 0.38443078
elapsed time for each run         : 10.950ms
```
