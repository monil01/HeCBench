using CUDA
include(joinpath(@__DIR__,"..","study_operations.jl"))
function source_kernel!(input,output,n::Int32)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 if i<n
  a=input[i+1];b=a+(isodd(i) ? UInt32(1) : typemax(UInt32));c=a⊻b;output[i+1]=simd_xor(a,b,c)
 end
 return
end
function probe_kernel!(input,output,n::Int32,p::Int32)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 if i<n
  a=input[3i+1];b=input[3i+2];c=input[3i+3];output[i+1]=individual_operation(i÷p,a,b,c)
 end
 return
end
function launch(input,output,n)
 @cuda threads=256 blocks=cld(n,256) source_kernel!(input,output,Int32(n))
end
function probe(input,output,n,p)
 @cuda threads=256 blocks=cld(n,256) probe_kernel!(input,output,Int32(n),Int32(p))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch,probe)
