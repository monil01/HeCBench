using CUDA,Libdl
h=CUDA.CUBLAS.handle()
math=Ref{UInt32}();CUDA.CUBLAS.cublasGetMathMode(h,math)
pointer=Ref{CUDA.CUBLAS.cublasPointerMode_t}();CUDA.CUBLAS.cublasGetPointerMode_v2(h,pointer)
path=realpath(Libdl.dlpath(CUDA.CUBLAS.libcublas))
println("VENDOR_POLICY {\"cublas_version\":\"$(CUDA.CUBLAS.version())\",\"path\":\"$path\",\"math_mode\":$(Int(math[])),\"pointer_mode\":$(Int(pointer[])),\"CUDA_jl_math_mode\":\"$(CUDA.math_mode())\",\"supplemental_environment_probe_not_primary_execution\":true}")
