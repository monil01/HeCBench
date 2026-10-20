using CUDA
include("../device_simulation.jl")
function simulate_kernel!(arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    gid=Int32((CUDA.blockIdx().x-1)*CUDA.blockDim().x+CUDA.threadIdx().x-1)
    simulate_gate!(gid,arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    nothing
end
function launch!(arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    @cuda threads=32 blocks=cld(n,Int32(32)) simulate_kernel!(arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    nothing
end
allocate_data(x)=CuArray(x)
allocate_arena(T,n)=CuArray{T}(undef,n)
include("../study_host.jl")
runstudy()
