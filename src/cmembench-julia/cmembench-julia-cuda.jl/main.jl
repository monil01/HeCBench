using CUDA
include("../study_constant.jl")
include("../study_common.jl")
function benchmark_constant!(out,v)
 result=constant_sum(v)
 if threadIdx().x==1&&blockIdx().x==1;@inbounds out[1]=result;end
 nothing
end
function factory(out,v)
 k=CUDA.@cuda launch=false benchmark_constant!(out,v);symbol=CUDA.CuGlobal{NTuple{1024,Int32}}(k.fun.mod,"study_constant_data")
 println("CMEM_PTX_BEGIN width=",sizeof(Int32)*typeof(v).parameters[1]);CUDA.code_ptx(stdout,benchmark_constant!,Tuple{typeof(cudaconvert(out)),typeof(v)};kernel=true,dump_module=true);println("CMEM_PTX_END")
 symbol,()->k(out,v;threads=256,blocks=16384)
end
constant_study(factory)
