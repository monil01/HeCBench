using CUDA,KernelAbstractions
include("../device_common.jl")
function allocate_model(a)
    out=KernelAbstractions.allocate(CUDABackend(),eltype(a),size(a))
    copyto!(out,a)
    out
end
@kernel function factor_kernel!(input,output,n)
    global_id=@index(Global,Linear)
    factor_one!(input,output,n,Int32(global_id)-Int32(1))
end
function launch_svd(w)
    factor_kernel!(CUDABackend(),256)(w.di,w.dout,Int32(w.n);ndrange=(w.n÷256+1)*256)
    nothing
end
include("../study_common.jl")
study_main("ka")
