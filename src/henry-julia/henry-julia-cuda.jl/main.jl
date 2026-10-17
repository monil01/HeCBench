using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(out,atoms,L)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 @inbounds out[i]=factor(sample(i-1,L)...,atoms,L,Val(false))
 return
end
run!(out,atoms,L)=(@cuda threads=256 blocks=16 kernel!(out,atoms,L))
study(run!)
