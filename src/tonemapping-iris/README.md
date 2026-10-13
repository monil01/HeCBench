IRIS DMem tone mapping port preserving image/repeat arguments, luminance and
gamma constants, warmups, default input path/repeat10000, and normalized-error
reference check. Input refresh explicitly updates DMem each run; output flushes.

The default ../tonemapping-sycl/input.hdr is absent locally. Validation uses a
documented positive RGB32x32 fixture and3repeats. Both native checks PASS and all
4096 finite output floats match HIP at rtol1e-6/atol1e-7. The optional
HECBENCH_DUMP_OUTPUT environment path writes raw float outputs for verification;
the isolated native validation source has the same observational hook.
No default dataset or large-repeat validation is claimed.
