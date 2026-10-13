IRIS dmem easyWave retaining original Okada-fault/grid preprocessing, four GPU
kernels, dynamic calculation window, open-boundary equations, source thresholds,
CLI/defaults and all Surfer/POI artifact writers. Host allocations and GPUlaunches
are IRIS dmem/tasks only. Node partial field/window writes use rw; readonly cache
arrays use r. Four dynamic bounds are updated after every host reset, and explicit
flush_out materializes updated bounds and node fields for POI/final outputs.
Float32/int32 scalar widths preserved; source kernels unchanged except exported
names. Tasks synchronize each original dependent phase (timing compromise).

Default -grid/-source/-time120 uses original easyWave-omp data paths. Validation
uses checksum-identical isolated copies of the existing original 27.9MB grid and
Bengkulu fault file, full default120-minute native/IRIS artifact checks plus a
bounded original CLI case with POI sampling. Native originals unchanged, no CPU
solver fallback or opaque vendor operation replacement. Actual HIP backend3/type256
AMD MI100 required. CUDA kernel source retained but NVIDIA unvalidated.
