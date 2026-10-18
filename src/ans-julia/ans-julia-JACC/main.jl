using CUDA,JACC
JACC.@init_backend
include("device.jl")
function entry_0(i,args...)
 gid=UInt32(i-1);width=UInt32(128);group=gid÷width
 ans_body(Val(0),args...,gid,group,width)
 nothing
end
function entry_1(i,args...)
 gid=UInt32(i-1);width=UInt32(128);group=gid÷width
 ans_body(Val(1),args...,gid,group,width)
 nothing
end
function entry_2(i,args...)
 gid=UInt32(i-1);width=UInt32(128);group=gid÷width
 ans_body(Val(2),args...,gid,group,width)
 nothing
end
function entry_3(i,args...)
 gid=UInt32(i-1);width=UInt32(128);group=gid÷width
 ans_body(Val(3),args...,gid,group,width)
 nothing
end
function entry_4(i,args...)
 gid=UInt32(i-1);width=UInt32(128);group=gid÷width
 ans_body(Val(4),args...,gid,group,width)
 nothing
end
function launch(op,args,a)
 f=getfield(@__MODULE__,Symbol(:entry_,op));N=Int(a[2])*128
 spec=JACC.launch_spec(;threads=128,blocks=Int(a[2]),shmem_size=0,stream=CUDA.legacy_stream(),sync=false)
 JACC.parallel_for(spec,N,f,args...)
 nothing
end
include("../study_common.jl")
main()
