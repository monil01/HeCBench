The original segmented sorting networks, bin histogram/grouping and 2048-item
block merge kernels are retained as device functions behind C-linkage IRIS
wrappers. All memory is DMem. Buffer swaps and two-dimensional grids follow
the native algorithm. Default 1<<25 elements, seed -278642091 and random
segment lengths0..5000 are unchanged. Both keys and key/value tests retain
the original host reference and duplicate-key value canonicalization.
SEGSORT_VALIDATION_ELEMENTS bounds only validation; SEGSORT_DUMP_DIR saves
canonical numeric outputs for exact native comparison.
CPU sorting references and artifact verification execute outside the GPU lock.
