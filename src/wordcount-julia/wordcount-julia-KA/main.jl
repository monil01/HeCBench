using CUDA
using KernelAbstractions
import Atomix
include(joinpath(@__DIR__,"study_kernel.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
const COUNTER=KernelAbstractions.allocate(BACKEND,Int32,1)
function allocate(x)
 d=KernelAbstractions.allocate(BACKEND,UInt8,length(x));copyto!(d,x);d
end
function countfn(d,n)
 n==0&&return 0
 fill!(COUNTER,0)
 count_ka!(BACKEND,128)(d,Int32(n),COUNTER;ndrange=cld(n,128)*128)
 Int(Array(COUNTER)[1])
end
study(allocate,countfn)
