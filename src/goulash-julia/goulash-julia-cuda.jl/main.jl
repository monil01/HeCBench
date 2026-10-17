using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(gate,voltage,n)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=n
 @inbounds gate[i]=step_gate(gate[i],voltage[i])
 end
 return
end
run!(g,v,n)=(@cuda threads=256 blocks=cld(n,256) kernel!(g,v,Int32(n));nothing)
study(run!)
