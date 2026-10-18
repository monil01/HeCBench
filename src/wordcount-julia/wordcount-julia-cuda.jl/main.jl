using CUDA
include(joinpath(@__DIR__,"..","study_reference_kernel.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const COUNTER=CUDA.zeros(Int32,1)
function countfn(d,n)
 n==0&&return 0
 fill!(COUNTER,0)
 CUDA.@cuda threads=128 blocks=cld(n,128) count_kernel!(d,Int32(n),COUNTER)
 Int(Array(COUNTER)[1])
end
study(x->CuArray(x),countfn)
