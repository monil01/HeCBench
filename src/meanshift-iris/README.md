IRIS DMem mean shift. File arguments, N=10000, D=3, M=3, all hyperparameters,
THREADS=64, default NUM_ITER=1000, and both original centroid checks are retained.
HECBENCH_MEANSHIFT_ITERATIONS optionally bounds validation iterations (default
1000). HECBENCH_DUMP optionally writes meanshift_base.csv/meanshift_opt.csv for
native numerical comparison. These options do not alter the default interface.
Two DMem buffers swap on-device with iris_r/iris_w. Separate host backing keeps
the original dataset unchanged. Between phases, the currently selected input
backing is restored and refreshed with DMem.update(host). Only final outputs
are flushed. Inactive tiling lanes use a valid row while retaining all barriers.
The supplied DVC dataset is absent locally; numerical validation uses a synthetic
10000-point three-cluster fixture and a test-only HIP adapter that adds the same
optional iteration/dump controls while retaining original native kernels/checks.
Original 1000-iteration dataset performance is not claimed as validated.
