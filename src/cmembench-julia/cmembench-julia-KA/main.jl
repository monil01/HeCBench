using CUDA,KernelAbstractions
include("../study_constant.jl")
include("../study_common.jl")
@kernel function benchmark_constant!(out,v)
 i=@index(Global,Linear)
 result=constant_sum(v)
 if i==1;@inbounds out[1]=result;end
end
function factory(out,v)
 backend=CUDABackend();obj=benchmark_constant!(backend,256)
 ndr,wgs,iterspace,dynamic=KernelAbstractions.launch_config(obj,4194304,nothing)
 ctx=KernelAbstractions.mkcontext(obj,ndr,iterspace)
 compiled=CUDA.@cuda launch=false always_inline=backend.always_inline maxthreads=256 obj.f(ctx,out,v)
 symbol=CUDA.CuGlobal{NTuple{1024,Int32}}(compiled.fun.mod,"study_constant_data")
 println("CMEM_PTX_BEGIN width=",sizeof(Int32)*typeof(v).parameters[1]);CUDA.code_ptx(stdout,obj.f,Tuple{typeof(cudaconvert(ctx)),typeof(cudaconvert(out)),typeof(v)};kernel=true,dump_module=true,maxthreads=256,always_inline=backend.always_inline);println("CMEM_PTX_END")
 symbol,()->obj(out,v;ndrange=4194304)
end
constant_study(factory)
