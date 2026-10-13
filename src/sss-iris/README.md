Recovered using actual GNU GSL 2.8, not a replacement library. Build with
make GSL=/tmp/hecbench-gsl-t1kQwR/install (GSL prefix is configurable).
Original host graph/search/scoring/integration/RNG routines and both device
kernels retained. Source RANDOMSEED=314159265 and run GSL_RNG_SEED=123,
full f9_n150_p50 data and all search defaults unchanged. IRIS initialization
occurs before GSL initialization; IRIS srand(time) does not alter the GSL RNG.

All device storage is DMem, updated after host packed-input changes. Tasks
preserve logical delete-before-add input dependency and flush integer results;
partial outputs use rw. No native host allocation/copy/launch.

P1: CanAddEdge shared workspace is static 16383 short elements, covering every
nonnegative original signed-short dynamic shared byte count (<=32766). Original
overflowing negative byte count is rejected rather than silently repaired.
Actual default p=50 remains unchanged. No graph/search algorithm replacement.
Prior missing-GSL deferral and all attempt history remain in local state.

DATA is read-only from the original sibling. The run target creates a DATA
symlink only when no DATA file/directory/link exists and never overwrites
inputs. DATA_DIR is configurable. Output stays in this port's RES directory.
Validation uses isolated run directories with the same DATA/RES convention.
