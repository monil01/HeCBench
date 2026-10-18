using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function lda_kernel!(cols,ptr,mask,counts,initialize::Bool,NW::Int32,D::Int32,K::Int32,iters::Int32,alpha,beta,gamma,grad,newbeta,trainloss,valloss,locks,grid::Int32,allocation::Val{SH}) where {SH}
 sh=@localmem(Float32,(SH,))
 BX=Int32(@index(Group,Linear))-Int32(1);TX=Int32(@index(Local,Linear))-Int32(1)
 lda_body!(cols,ptr,mask,counts,initialize,NW,D,K,iters,alpha,beta,gamma,grad,newbeta,trainloss,valloss,locks,sh,BX,TX,grid)
end
function launch(c,initialize)
 I,F,M=c.iv,c.fv,c.mv
 lda_kernel!(BACKEND,256)(I[1],I[2],M,F[1],initialize,Int32(c.NW),Int32(c.D),Int32(c.K),Int32(c.iters),F[2],F[3],F[4],F[5],F[6],F[7],F[8],I[3],Int32(c.blocks),Val(4*c.K+32);ndrange=256*c.blocks)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
