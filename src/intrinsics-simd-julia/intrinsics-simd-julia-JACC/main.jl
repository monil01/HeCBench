using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_operations.jl"))
function source_kernel!(xx,input,output,n::Int32)
 i=Int32(xx)-Int32(1)
 if i<n
  a=input[i+1];b=a+(isodd(i) ? UInt32(1) : typemax(UInt32));c=a⊻b;output[i+1]=simd_xor(a,b,c)
 end
 return nothing
end
function probe_kernel!(xx,input,output,n::Int32,p::Int32)
 i=Int32(xx)-Int32(1)
 if i<n
  a=input[3i+1];b=input[3i+2];c=input[3i+3];output[i+1]=individual_operation(i÷p,a,b,c)
 end
 return nothing
end
function launch(input,output,n)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,n,source_kernel!,input,output,Int32(n))
end
function probe(input,output,n,p)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,n,probe_kernel!,input,output,Int32(n),Int32(p))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch,probe)
