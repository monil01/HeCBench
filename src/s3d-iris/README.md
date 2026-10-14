S3D retains all 27 native reaction kernels and both SP/DP pipelines.
Device template bodies are unchanged behind C-linkage IRIS wrappers.
Original option parser, default passes10, make run passes100, size classes,
block sizes64/128 and mass-fraction/temperature/pressure initialization remain.
All eleven device arrays use DMem. S3D_DUMP_DIR optionally exports every
22-species output at every grid point for native comparison outside GPU lock.
Validation uses -q -n 2 -s 1: full original size1 grid and complete repeated
SP/DP pipelines. CPU artifact comparison checks finite outputs with explicit
SP/DP tolerances, rather than accepting timing output.
