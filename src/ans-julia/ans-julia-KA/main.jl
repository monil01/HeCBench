using CUDA,KernelAbstractions
include("device.jl")
@kernel unsafe_indices=true function entry_0(IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes)
 width=UInt32(128);group=UInt32(@index(Group,Linear))-UInt32(1);localidx=UInt32(@index(Local,Linear))-UInt32(1);gid=group*width+localidx
 ans_body(Val(0),IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes,gid,group,width)
end
@kernel unsafe_indices=true function entry_1(IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes)
 width=UInt32(128);group=UInt32(@index(Group,Linear))-UInt32(1);localidx=UInt32(@index(Local,Linear))-UInt32(1);gid=group*width+localidx
 ans_body(Val(1),IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes,gid,group,width)
end
@kernel unsafe_indices=true function entry_2(IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes)
 width=UInt32(128);group=UInt32(@index(Group,Linear))-UInt32(1);localidx=UInt32(@index(Local,Linear))-UInt32(1);gid=group*width+localidx
 ans_body(Val(2),IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes,gid,group,width)
end
@kernel unsafe_indices=true function entry_3(IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes)
 width=UInt32(128);group=UInt32(@index(Group,Linear))-UInt32(1);localidx=UInt32(@index(Local,Linear))-UInt32(1);gid=group*width+localidx
 ans_body(Val(3),IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes,gid,group,width)
end
@kernel unsafe_indices=true function entry_4(IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes)
 width=UInt32(128);group=UInt32(@index(Group,Linear))-UInt32(1);localidx=UInt32(@index(Local,Linear))-UInt32(1);gid=group*width+localidx
 ans_body(Val(4),IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes,gid,group,width)
end
function launch(op,args,a)
 f=getfield(@__MODULE__,Symbol(:entry_,op));kernel=f(CUDABackend(),128)
 CUDA.stream!(CUDA.legacy_stream()) do
  kernel(args...;ndrange=Int(a[2])*128)
 end
 nothing
end
include("../study_common.jl")
main()
