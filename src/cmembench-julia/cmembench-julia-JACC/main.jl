using CUDA,JACC
JACC.@init_backend
include("../study_constant.jl")
include("../study_common.jl")
function benchmark_constant!(i,out,v)
 result=constant_sum(v)
 if i==1;@inbounds out[1]=result;end
 nothing
end
function factory(out,v)
 ext=Base.get_extension(JACC,:CUDAExt)
 spec=JACC.launch_spec(;threads=256,blocks=16384,shmem_size=0,sync=false)
 kargs=ext._kernel_args(4194304,benchmark_constant!,out,v)
 compiled,_=ext._kernel_maxshmem(ext._parallel_for_cuda,kargs,nothing)
 symbol=CUDA.CuGlobal{NTuple{1024,Int32}}(compiled.fun.mod,"study_constant_data")
 println("CMEM_PTX_BEGIN width=",sizeof(Int32)*typeof(v).parameters[1]);CUDA.code_ptx(stdout,ext._parallel_for_cuda,Tuple{Core.Typeof.(kargs)...};kernel=true,dump_module=true);println("CMEM_PTX_END")
 symbol,()->JACC.parallel_for(spec,4194304,benchmark_constant!,out,v)
end
constant_study(factory)
