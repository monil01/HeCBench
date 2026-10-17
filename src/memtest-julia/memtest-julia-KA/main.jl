using CUDA, KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
@kernel function init!(buf,n)
 body_init!(UInt64(@index(Global,Linear))-UInt64(1),UInt64(4194304),buf,n)
end
@kernel function move!(buf,n)
 body_move!(UInt64(@index(Global,Linear))-UInt64(1),UInt64(4194304),buf,n)
end
@kernel function check!(buf,n,count,addr,expect,current,second)
 body_check!(UInt64(@index(Global,Linear))-UInt64(1),UInt64(4194304),buf,n,count,addr,expect,current,second)
end
function run!(which,args...)
 f=which==:init ? init! : which==:move ? move! : check!
 f(get_backend(args[1]),64)(args...;ndrange=4194304)
 return
end
study(run!)
