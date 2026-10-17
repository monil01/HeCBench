using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(a,b,c,n)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);add_body(i,a,b,c,n);return
end
run!(a,b,c,n)=(@cuda threads=256 blocks=cld(n,256) kernel!(a,b,c,n);nothing)
study(run!)
