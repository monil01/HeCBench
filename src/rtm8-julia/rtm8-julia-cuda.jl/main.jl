using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(v,cs,formalcr,formalns,nr,image,a)
 x=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);y=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y-Int32(1);z=blockIdx().z-Int32(1)
 body(x,y,z,v,cs,formalcr,formalns,nr,image,a);return
end
run!(v,cs,ns,cr,nr,image,a)=(@cuda threads=(16,16,1) blocks=(cld(NX,16),cld(NY,16),NZ) kernel!(v,cs,ns,cr,nr,image,a);nothing)
study(run!)
