using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
function init!(i,buf,n)
 body_init!(UInt64(i)-UInt64(1),UInt64(4194304),buf,n);return
end
function move!(i,buf,n)
 body_move!(UInt64(i)-UInt64(1),UInt64(4194304),buf,n);return
end
function check!(i,buf,n,count,addr,expect,current,second)
 body_check!(UInt64(i)-UInt64(1),UInt64(4194304),buf,n,count,addr,expect,current,second);return
end
function run!(which,args...)
 f=which==:init ? init! : which==:move ? move! : check!
 JACC.parallel_for(JACC.launch_spec(threads=64,blocks=65536,shmem_size=0,sync=false),4194304,f,args...)
 return
end
study(run!)
