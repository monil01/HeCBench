using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
const SH7=JACC.zeros(Float32,60)
const SH33=JACC.zeros(Float32,164)
const SH65=JACC.zeros(Float32,292)
function lda_kernel!(xx,cols,ptr,mask,counts,initialize::Bool,NW::Int32,D::Int32,K::Int32,iters::Int32,alpha,beta,gamma,grad,newbeta,trainloss,valloss,locks,grid::Int32,template)
 sh=JACC.shared(template);tid=Int32(xx)-Int32(1);BX=tid÷Int32(256);TX=tid%Int32(256)
 lda_body!(cols,ptr,mask,counts,initialize,NW,D,K,iters,alpha,beta,gamma,grad,newbeta,trainloss,valloss,locks,sh,BX,TX,grid)
 return nothing
end
function launch(c,initialize)
 I,F,M=c.iv,c.fv,c.mv;template=c.K==7 ? SH7 : (c.K==33 ? SH33 : SH65)
 spec=JACC.launch_spec(;threads=256,blocks=c.blocks,shmem_size=4*(4*c.K+32),sync=false)
 JACC.parallel_for(spec,256*c.blocks,lda_kernel!,I[1],I[2],M,F[1],initialize,Int32(c.NW),Int32(c.D),Int32(c.K),Int32(c.iters),F[2],F[3],F[4],F[5],F[6],F[7],F[8],I[3],Int32(c.blocks),template)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
