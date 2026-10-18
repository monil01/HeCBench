include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(variant,A,R,P,n,M)
 if variant==1;@cuda threads=192 blocks=cld(n,Int32(192)) aos_cuda!(A,R,P,n,M)
 elseif variant==2;@cuda threads=192 blocks=cld(n,Int32(192)) soa_cuda!(A,R,P,n,M)
 else;@cuda threads=96 blocks=cld(n,Int32(96)) shmem=32256 shared_cuda!(A,R,P,n,M);end
 return
end
study(launch,(T,n)->CuArray{T}(undef,n))
