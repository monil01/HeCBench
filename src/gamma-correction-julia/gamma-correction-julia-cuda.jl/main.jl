using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(pixel,n)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=n
 @inbounds pixel[i]=transform(pixel[i])
 end
 return
end
run!(p,n)=(@cuda threads=256 blocks=cld(n,256) kernel!(p,Int32(n));nothing)
study(run!)
