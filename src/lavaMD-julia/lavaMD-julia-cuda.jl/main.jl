using CUDA
include(joinpath(@__DIR__,"..","study_types.jl"))
include(joinpath(@__DIR__,"..","study_body.jl"))
function md_kernel!(meta,rv,q,fv,alpha::Float32,nb::Int32)
 sh=CuStaticSharedArray(Float32,900)
 BX=Int32(blockIdx().x)-Int32(1);TX=Int32(threadIdx().x)-Int32(1)
 md_body!(meta,rv,q,fv,alpha,nb,BX,TX,sh,CUDA.sync_threads)
 return
end
function launch(meta,rv,q,fv,alpha,nb)
 @cuda threads=128 blocks=nb md_kernel!(meta,rv,q,fv,alpha,Int32(nb))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
