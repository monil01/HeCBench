# nn IRIS DMem port

CUDA argument parsing, defaults, 64-thread geometry, Euclidean float distances and top-N selection are retained. Memory belongs to IRIS DMem; HIP GPU selection is required (no CPU fallback). Build with make; run via IRIS_ARCHS=HIP make run RUN_ARGS='filelist.txt -r 5 -lat 30 -lng 90 -i 3 -t'.

No input database/filelist is supplied by the source benchmark. Provide the original fixed-width records: 48 characters plus newline, latitude at offset28/width5, longitude at offset33/width5, and filelist entries naming database files. The default run's repeat10000 is unchanged. Invalid missing filelist now reports an error instead of dereferencing null.

Set HECBENCH_DUMP=1 to write the complete pre-selection float distances to nn_distances.bin for numeric comparison; no input or kernel changes. Native HIP test-only adapter uses the same dump. Verification uses deterministic 129/256-record fixtures (partial/full workgroups), repeats1/3, full distances against HIP and an independent Euclidean reference, plus the printed top-five records.

The original findLowest record-swap aliasing defect is intentionally preserved for source-faithful native output comparison. Distances are separately verified before that legacy selection mutates the arrays. Timing includes IRIS task overhead and is not correctness evidence.
