using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function entry_1!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L::Int32,M::Int32,N::Int32)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1);j=(Int32(blockIdx().y)-Int32(1))*Int32(blockDim().y)+Int32(threadIdx().y)-Int32(1);k=(Int32(blockIdx().z)-Int32(1))*Int32(blockDim().z)+Int32(threadIdx().z)-Int32(1)
 hypterm_body_1!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L,M,N,i,j,k)
 return
end
function entry_2!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L::Int32,M::Int32,N::Int32)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1);j=(Int32(blockIdx().y)-Int32(1))*Int32(blockDim().y)+Int32(threadIdx().y)-Int32(1);k=(Int32(blockIdx().z)-Int32(1))*Int32(blockDim().z)+Int32(threadIdx().z)-Int32(1)
 hypterm_body_2!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L,M,N,i,j,k)
 return
end
function entry_3!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L::Int32,M::Int32,N::Int32)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1);j=(Int32(blockIdx().y)-Int32(1))*Int32(blockDim().y)+Int32(threadIdx().y)-Int32(1);k=(Int32(blockIdx().z)-Int32(1))*Int32(blockDim().z)+Int32(threadIdx().z)-Int32(1)
 hypterm_body_3!(flux_0,flux_1,flux_2,flux_3,flux_4,cons_1,cons_2,cons_3,cons_4,q_1,q_2,q_3,q_4,dxinv0,dxinv1,dxinv2,L,M,N,i,j,k)
 return
end
const ENTRIES=(entry_1!,entry_2!,entry_3!)
function launch(stage,flux,cons,q,dx,n)
 k=ENTRIES[stage];@cuda threads=(16,4,4) blocks=(cld(n,16),cld(n,4),cld(n,4)) k(flux...,cons...,q...,dx...,Int32(n),Int32(n),Int32(n))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
