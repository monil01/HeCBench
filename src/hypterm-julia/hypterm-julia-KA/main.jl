using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function entry_1!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L::Int32,M::Int32,N::Int32)
 t=@index(Global,Cartesian);i=Int32(t[1])-Int32(1);j=Int32(t[2])-Int32(1);k=Int32(t[3])-Int32(1)
 hypterm_body_1!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L,M,N,i,j,k)
end
@kernel function entry_2!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L::Int32,M::Int32,N::Int32)
 t=@index(Global,Cartesian);i=Int32(t[1])-Int32(1);j=Int32(t[2])-Int32(1);k=Int32(t[3])-Int32(1)
 hypterm_body_2!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L,M,N,i,j,k)
end
@kernel function entry_3!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L::Int32,M::Int32,N::Int32)
 t=@index(Global,Cartesian);i=Int32(t[1])-Int32(1);j=Int32(t[2])-Int32(1);k=Int32(t[3])-Int32(1)
 hypterm_body_3!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L,M,N,i,j,k)
end
const ENTRIES=(entry_1!,entry_2!,entry_3!)
function launch(stage,flux,cons,q,dx,n)
 k=ENTRIES[stage](BACKEND,(16,4,4));k(flux...,cons...,q...,dx...,Int32(n),Int32(n),Int32(n);ndrange=(cld(n,16)*16,cld(n,4)*4,cld(n,4)*4))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
