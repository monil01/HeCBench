using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function entry_1!(xx,yy,zz,flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L::Int32,M::Int32,N::Int32)
 i=Int32(xx)-Int32(1);j=Int32(yy)-Int32(1);k=Int32(zz)-Int32(1)
 hypterm_body_1!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L,M,N,i,j,k)
 return nothing
end
function entry_2!(xx,yy,zz,flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L::Int32,M::Int32,N::Int32)
 i=Int32(xx)-Int32(1);j=Int32(yy)-Int32(1);k=Int32(zz)-Int32(1)
 hypterm_body_2!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L,M,N,i,j,k)
 return nothing
end
function entry_3!(xx,yy,zz,flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L::Int32,M::Int32,N::Int32)
 i=Int32(xx)-Int32(1);j=Int32(yy)-Int32(1);k=Int32(zz)-Int32(1)
 hypterm_body_3!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L,M,N,i,j,k)
 return nothing
end
const ENTRIES=(entry_1!,entry_2!,entry_3!)
function launch(stage,flux,cons,q,dx,n)
 spec=JACC.launch_spec(;threads=(16,4,4),blocks=(cld(n,16),cld(n,4),cld(n,4)),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(n,n,n),ENTRIES[stage],flux...,cons...,q...,dx...,Int32(n),Int32(n),Int32(n))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
