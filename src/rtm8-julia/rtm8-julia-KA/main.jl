using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(v,cs,formalcr,formalns,nr,image,a)
 idx=@index(Global,NTuple)
 body(Int32(idx[1])-Int32(1),Int32(idx[2])-Int32(1),Int32(idx[3])-Int32(1),v,cs,formalcr,formalns,nr,image,a)
end
run!(v,cs,ns,cr,nr,image,a)=(kernel!(get_backend(image),(16,16,1))(v,cs,ns,cr,nr,image,a;ndrange=(cld(NX,16)*16,cld(NY,16)*16,NZ));nothing)
study(run!)
