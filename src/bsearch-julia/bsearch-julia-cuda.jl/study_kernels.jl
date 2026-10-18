include(joinpath(@__DIR__,"..","search_math.jl"))
function bs_bit_kernel!(a,z,r,qn::Int64,n::Int64,mode::Val{MODE}) where {MODE}
 i=Int64((blockIdx().x-1)*blockDim().x+threadIdx().x)
 if i<=qn;@inbounds r[i]=search_walk(a,z[i],n,search_seed(n),mode);end
 return
end
function bs_shared_kernel!(a,z,r,qn::Int64,n::Int64)
 i=Int64((blockIdx().x-1)*blockDim().x+threadIdx().x)
 seed=CUDA.@cuStaticSharedMem(UInt64,1)
 if threadIdx().x==1;@inbounds seed[1]=search_seed(n);end
 CUDA.sync_threads()
 if i<=qn;@inbounds r[i]=search_walk(a,z[i],n,seed[1],Val(3));end
 return
end
