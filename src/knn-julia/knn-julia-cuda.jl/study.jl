include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function knn_call!(a,b,n,q,d,k,hd,hi)
 da=CuArray(a);db=CuArray(b);ds=CuArray{Float32}(undef,n*q);di=CuArray{Int32}(undef,k*q)
 @cuda threads=(16,16) blocks=(cld(q,16),cld(n,16)) distance_kernel!(da,n,db,q,d,ds)
 @cuda threads=256 blocks=cld(q,256) sort_kernel!(ds,di,q,n,k)
 @cuda threads=(16,16) blocks=(cld(q,16),cld(k,16)) sqrt_kernel!(ds,q,k)
 CUDA.synchronize();copyto!(hd,1,ds,1,k*q);copyto!(hi,di)
 for a in (da,db,ds,di);CUDA.unsafe_free!(a);end
 nothing
end
study_main(knn_call!)
