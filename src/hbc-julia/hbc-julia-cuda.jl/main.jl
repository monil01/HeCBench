using CUDA
const MODEL="cuda_jl"
include("../study_device.jl")
function hbc_kernel!(d,p,n,m)
 sh=CUDA.CuStaticSharedArray(Int32,528)
 tid=CUDA.threadIdx().x-Int32(1);block=CUDA.blockIdx().x-Int32(1)
 hbc_body!(d,p,n,m,sh,tid,block,Val(:cuda_jl));nothing
end
function launch_hbc(s)
 CUDA.@cuda maxregs=64 threads=1024 blocks=80 hbc_kernel!(s.device,s.strides,Int32(s.n),Int32(s.m))
 nothing
end
include("../study_common.jl")
main()
