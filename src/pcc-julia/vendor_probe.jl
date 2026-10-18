using CUDA,Libdl,SHA
function make_study_cublas()
 h=Ref{CUDA.CUBLAS.cublasHandle_t}();CUDA.CUBLAS.cublasCreate_v2(h)
 CUDA.CUBLAS.cublasSetMathMode(h[],CUDA.CUBLAS.CUBLAS_DEFAULT_MATH)
 mode=Ref{UInt32}();CUDA.CUBLAS.cublasGetMathMode(h[],mode)
 pointermode=Ref{CUDA.CUBLAS.cublasPointerMode_t}();CUDA.CUBLAS.cublasGetPointerMode_v2(h[],pointermode);@assert pointermode[]==0
 version=Ref{Cint}();CUDA.CUBLAS.cublasGetVersion_v2(h[],version)
 paths=filter(x->occursin(r"libcublas\.so",basename(x)),Libdl.dllist());actual=unique(realpath.(paths));@assert length(actual)==1
 path=only(actual);digest=bytes2hex(sha256(read(path)));@assert path=="/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224"&&digest=="6571bf8344cf13f3b116e6d613dd234cefca081c2303c3667e71b25d0eba5ec2"&&Int(mode[])==0
 println("STUDY_VENDOR {\"cublas_realpath\":\"$path\",\"sha256\":\"$digest\",\"version_integer\":$(version[]),\"math_mode\":$(Int(mode[])),\"typed_api\":\"cublasSgemm_v2\",\"own_handle\":true}")
 h[]
end
