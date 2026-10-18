using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_operations.jl"))
@kernel function source_kernel!(input,output,n::Int32)
 i=Int32(@index(Global,Linear))-Int32(1)
 if i<n
  a=input[i+1];b=a+(isodd(i) ? UInt32(1) : typemax(UInt32));c=a⊻b;output[i+1]=simd_xor(a,b,c)
 end
end
@kernel function probe_kernel!(input,output,n::Int32,p::Int32)
 i=Int32(@index(Global,Linear))-Int32(1)
 if i<n
  a=input[3i+1];b=input[3i+2];c=input[3i+3];output[i+1]=individual_operation(i÷p,a,b,c)
 end
end
function launch(input,output,n)
 source_kernel!(BACKEND,256)(input,output,Int32(n);ndrange=cld(n,256)*256)
end
function probe(input,output,n,p)
 probe_kernel!(BACKEND,256)(input,output,Int32(n),Int32(p);ndrange=cld(n,256)*256)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch,probe)
