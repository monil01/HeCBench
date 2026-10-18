using CUDA,JACC
JACC.@init_backend
include("device.jl")
function entry_0(i,j,args...)
 snicit_body(Val(0),args...)
 nothing
end
function entry_1(i,j,args...)
 snicit_body(Val(1),args...)
 nothing
end
function entry_2(i,j,args...)
 snicit_body(Val(2),args...)
 nothing
end
function entry_3(i,j,args...)
 snicit_body(Val(3),args...)
 nothing
end
function entry_4(i,j,args...)
 snicit_body(Val(4),args...)
 nothing
end
function entry_5(i,j,args...)
 snicit_body(Val(5),args...)
 nothing
end
function entry_6(i,j,args...)
 snicit_body(Val(6),args...)
 nothing
end
function entry_7(i,j,args...)
 snicit_body(Val(7),args...)
 nothing
end
function entry_8(i,j,args...)
 snicit_body(Val(8),args...)
 nothing
end
function launch(op,args,a,stream)
 f=getfield(@__MODULE__,Symbol(:entry_,op));bx=Int(a[6]);by=Int(a[7]);groups=Int(a[5])
 spec=JACC.launch_spec(;threads=(bx,by),blocks=(groups,1),shmem_size=Int(a[8]),stream=stream,sync=false)
 JACC.parallel_for(spec,(groups*bx,by),f,args...)
 nothing
end
include("../study_common.jl")
main()
