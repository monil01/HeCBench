# STSG IRIS dmem port

Retains all four original NDVI/QA conversion, reference generation, neighborhood
comparison and Savitzky-Golay filtering device kernels, configuration parser,
scalar types, 16x16 geometry, memory-based row batching, and source checksum.
All host device-buffer ownership/copies/flushes use iris::DMem; partial writes
use iris_rw and host resets call update after preserving other device-written data.
The only HIP runtime host call is a read-only hipMemGetInfo metadata query via
dlopen, preserving the original min(totalGlobalMem,4GiB) batching budget.
No native allocations/copies/launches or vendor-operation replacement.

make GDAL_INCLUDE=/path/to/include GDAL_LIB=-l:libgdal.so.34 builds using real GDAL;
IRIS_ROOT defaults /tmp/iris. make run preserves original example.txt config CLI.
Provide actual input prefixes in the parameter file. Host GDAL raster IO remains
unchanged; STSG_Test_path is parsed but not written in this HeCBench source.
HECBENCH_OUTPUT optionally dumps all final float outputs before unchanged checksum.

Validation uses original upstream commit877fa91b5f2f8886618bb0380cb5627c5653949f
NDVI/Reliability2001..2003 GeoTIFF crops64x64x23bands, cropwindow256,256,64,64,
with original cosyear=.90,win_year=2,win=10,sampcorr=.9,snow_address=1.
GDAL3.8.4 headers and real installed runtime, no system install.
Isolated native adapter is capture-only; all original algorithms/defaults unchanged.
This is bounded actual-data validation, not full-production1200x1200/18-year validation.
kernel.cu retains original CUDA device arithmetic, not locally CUDA-validated.
