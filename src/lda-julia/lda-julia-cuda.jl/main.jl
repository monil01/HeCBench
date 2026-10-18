using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function lda_kernel!(cols,ptr,mask,counts,initialize::Bool,NW::Int32,D::Int32,K::Int32,iters::Int32,alpha,beta,gamma,grad,newbeta,trainloss,valloss,locks)
 sh=CuDynamicSharedArray(Float32,4*K+32)
 lda_body!(cols,ptr,mask,counts,initialize,NW,D,K,iters,alpha,beta,gamma,grad,newbeta,trainloss,valloss,locks,sh,Int32(blockIdx().x)-Int32(1),Int32(threadIdx().x)-Int32(1),Int32(gridDim().x))
 return
end
function launch(c,initialize)
 I,F,M=c.iv,c.fv,c.mv
 @cuda threads=256 blocks=c.blocks shmem=4*(4*c.K+32) lda_kernel!(I[1],I[2],M,F[1],initialize,Int32(c.NW),Int32(c.D),Int32(c.K),Int32(c.iters),F[2],F[3],F[4],F[5],F[6],F[7],F[8],I[3])
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
