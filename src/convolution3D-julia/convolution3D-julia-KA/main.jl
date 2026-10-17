using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(x,w,y,C,M,K,Hin,Win,Hout,Wout,Wgrid,mode)
 localidx=@index(Local,NTuple);group=@index(Group,NTuple)
 conv_body(Int32(localidx[1])-Int32(1),Int32(localidx[2])-Int32(1),Int32(group[1])-Int32(1),Int32(group[2])-Int32(1),Int32(group[3])-Int32(1),x,w,y,C,M,K,Hin,Win,Hout,Wout,Wgrid,mode)
end
function run!(x,w,y,dims,::Val{STYLE}) where STYLE
 g=grid(STYLE);kernel!(get_backend(y),(16,16,1))(x,w,y,dims...,Val(STYLE);ndrange=(16g[1],16g[2],g[3]))
 return
end
study(run!)
