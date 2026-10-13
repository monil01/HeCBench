IRIS DMem molecular dynamics preserves single-precision default SP=yes, optional
double SP=no, class selectors{12288,24576,36864,73728}, seed123,128 nearest neighbors,
cutoff13.5,LJ1.5/2.0,domain20,iteration argument and default3 1000. HIP/host
float4 anddouble4 share16-byte alignment and16/32-byte size. Original neighbor
builder, finite-result asserts and max-error CPU summary retained.
HECBENCH_BOUNDED_VALIDATION compile flag selects129,257,513,1025 atoms (all>128)
in both port and local native driver, preserving default selector behavior otherwise.
Common native/IRIS numerical bound: finite max error<=1e-5 for tested float mode,
with summary difference<=1e-7. DMem iris_rw buffers; explicit input updates/output flush.
CUDA kernel unvalidated; timings include IRIS overhead.
