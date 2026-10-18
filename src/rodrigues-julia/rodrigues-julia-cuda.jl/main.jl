using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function rotate_cuda!(d,p,n::Int32,width)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n;@inbounds rotate_point!(d,i,p[5],p[6],p[7],p[8],width);end
 return
end
function factory(d,p,width)
 n=Int32(RN)
 kernel=@cuda launch=false rotate_cuda!(d,p,n,width)
 n->kernel(d,p,n,width;threads=256,blocks=cld(n,256))
end
rodrigues_study(factory)
