using CUDA,KernelAbstractions
include("../study_device.jl")
@kernel function initkernel(A,n,dim,seed,threshold)
 tid=@index(Local,Linear);bid=@index(Group,Linear)
 initbody(A,n,dim,seed,threshold,Int32(tid)-1,Int32(bid)-1)
end
@kernel function distancekernel(A,B,C,h,w,pw,weight,E)
 tid=@index(Local,Linear);bid=@index(Group,Linear)
 distbody(A,B,C,h,w,pw,weight,E,Int32(tid)-1,Int32(bid)-1)
end
@kernel function rowkernel(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight)
 tid=@index(Local,Linear);bid=@index(Group,Linear)
 updatebody(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight,Int32(tid)-1,Int32(bid)-1,Val(false))
end
@kernel function colkernel(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight)
 tid=@index(Local,Linear);bid=@index(Group,Linear)
 updatebody(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight,Int32(tid)-1,Int32(bid)-1,Val(true))
end
launchinit(A,n,dim,seed,threshold)=initkernel(CUDABackend(),512)(A,n,dim,seed,threshold;ndrange=cld(n,512)*512)
launchdist(A,B,C,h,w,pw,weight,E)=distancekernel(CUDABackend(),512)(A,B,C,h,w,pw,weight,E;ndrange=cld(h,16)*512)
function launchupdate(col,args...)
 n=col ? args[5] : args[4];blocks=cld(min(112,n),16)
 kernel=col ? colkernel : rowkernel
 kernel(CUDABackend(),512)(args...;ndrange=blocks*512)
 nothing
end
include("../study_common.jl")
main()
