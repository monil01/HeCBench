QKV IRIS DMem port
==================
All four original CLI variants and default Make run sequence are retained.
Defaults B=4,T=1024,C=768,OC=2304, seed=0, repeats=100, tolerance=0.1.
Optional HECBENCH_QKV_{B,T,C,OC,REPEAT} enable bounded validation without changing defaults.
Kernel 4 retains the original 128x128 tile and requires BT,OC multiples of128 and C multiple of32; unsupported shapes fail explicitly.
Variants 2/3 lower BLAS/Lt GEMM to a portable FP32 tiled HIP kernel submitted through IRIS: variant2 retains separate bias, variant3 fuses bias. No GELU (original has_gelu=0). This is a documented performance compromise, not a BLAS speed comparison.
The host owns only DMem; outputs flush to host; bias updates preserve intermediate device data using iris_rw. HIP-only backend selection is checked.
Optional HECBENCH_QKV_OUTPUT dumps all verified floats for native/IRIS artifact comparison.
