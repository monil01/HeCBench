using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function cast_from_jacc!(i,input,out,n::Int32)
 i=Int32(i)
 if i<=n;@inbounds out[i]=cast_from_fp(input[i]);end
 return
end
function cast_to_jacc!(i,input,out,n::Int32)
 i=Int32(i)
 if i<=n;@inbounds out[i]=cast_to_fp(input[i]);end
 return
end
function factory(input,out,n,mode)
 k=mode=="from_fp" ? cast_from_jacc! : cast_to_jacc!
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 return ()->JACC.parallel_for(spec,n,k,input,out,Int32(n))
end
cast_study(factory)
