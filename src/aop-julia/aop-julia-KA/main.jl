using CUDA, KernelAbstractions
import Atomix
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
@inline function aop_ka_sync()
 @synchronize
 nothing
end
@kernel function path_kernel(s,p,n,steps,K,S0,r,sigma,dt,put)
 g=Int32(@index(Global,Linear))-Int32(1)
 paths_body!(s,p,n,steps,K,S0,r,sigma,dt,put,g)
 nothing
end
@kernel function prepare_kernel(p,f,v,n,K,put)
 ints=@localmem Int32 (514,);mem=@localmem Float64 (16,)
 prepare_body!(p,f,v,n,K,put,Int32(@index(Group,Linear))-Int32(1),Int32(@index(Local,Linear))-Int32(1),ints,mem,aop_ka_sync)
 nothing
end
@kernel function partial_beta_kernel(p,f,v,t,n,step,K,put)
 mem=@localmem Float64 (15,)
 partial_beta_body!(p,f,v,t,n,step,K,put,Int32(@index(Group,Linear))-Int32(1),Int32(@index(Local,Linear))-Int32(1),mem,aop_ka_sync)
 nothing
end
@kernel function final_beta_kernel(f,t,step)
 mem=@localmem Float64 (3,)
 final_beta_body!(f,t,step,Int32(@index(Local,Linear))-Int32(1),mem,aop_ka_sync)
 nothing
end
@kernel function cash_kernel(p,f,t,n,step,K,put,discount)
 g=Int32(@index(Global,Linear))-Int32(1);stride=Int32(@groupsize()[1]*@ndrange()[1]÷@groupsize()[1])
 cash_body!(p,f,t,n,step,K,put,discount,g,stride)
 nothing
end
@kernel function partial_sum_kernel(p,t,n)
 mem=@localmem Float64 (1,)
 partial_sum_body!(p,t,n,Int32(@index(Group,Linear))-Int32(1),Int32(@index(Local,Linear))-Int32(1),mem,aop_ka_sync)
 nothing
end
@kernel function final_sum_kernel(t,n,blocks,discount)
 mem=@localmem Float64 (1,)
 final_sum_body!(t,n,blocks,discount,Int32(@index(Local,Linear))-Int32(1),mem,aop_ka_sync)
 nothing
end
function launch(kind,args...;threads,blocks)
 if kind==1;path_kernel(KernelAbstractions.get_backend(args[1]),threads)(args...;ndrange=threads*blocks)
 elseif kind==2;prepare_kernel(KernelAbstractions.get_backend(args[1]),threads)(args...;ndrange=threads*blocks)
 elseif kind==3;partial_beta_kernel(KernelAbstractions.get_backend(args[1]),threads)(args...;ndrange=threads*blocks)
 elseif kind==4;final_beta_kernel(KernelAbstractions.get_backend(args[1]),threads)(args...;ndrange=threads*blocks)
 elseif kind==5;cash_kernel(KernelAbstractions.get_backend(args[1]),threads)(args...;ndrange=threads*blocks)
 elseif kind==6;partial_sum_kernel(KernelAbstractions.get_backend(args[1]),threads)(args...;ndrange=threads*blocks)
 else;final_sum_kernel(KernelAbstractions.get_backend(args[1]),threads)(args...;ndrange=threads*blocks)
 end
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
function allocate(v)
 a=KernelAbstractions.allocate(CUDA.CUDABackend(),eltype(v),size(v));copyto!(a,v);a
end
study(allocate,launch)
