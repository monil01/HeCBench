using CUDA,KernelAbstractions
include("../device_simulation.jl")
@kernel function simulate_kernel!(arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    gid=Int32(@index(Global,Linear))-Int32(1)
    simulate_gate!(gid,arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
end
function launch!(arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    simulate_kernel!(CUDABackend(),32)(arena,func,input,delay,output,capacity,sizes,flags,truth,n,off;ndrange=Int(cld(n,Int32(32))*Int32(32)))
    nothing
end
allocate_data(x)=begin
    a=KernelAbstractions.allocate(CUDABackend(),eltype(x),size(x))
    copyto!(a,x)
    a
end
allocate_arena(T,n)=KernelAbstractions.allocate(CUDABackend(),T,(n,))
include("../study_host.jl")
runstudy()
