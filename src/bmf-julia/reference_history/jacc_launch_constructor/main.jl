using CUDA,JACC
include("../study_device.jl")
function initkernel(i,A,n,dim,seed,threshold)
 initbody(A,n,dim,seed,threshold,Int32((i-1)%512),Int32((i-1)÷512));nothing
end
function distancekernel(i,A,B,C,h,w,pw,weight,E)
 distbody(A,B,C,h,w,pw,weight,E,Int32((i-1)%512),Int32((i-1)÷512));nothing
end
function rowkernel(i,A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight)
 updatebody(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight,Int32((i-1)%512),Int32((i-1)÷512),Val(false));nothing
end
function colkernel(i,A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight)
 updatebody(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight,Int32((i-1)%512),Int32((i-1)÷512),Val(true));nothing
end
function jlaunch(f,blocks,args...)
 spec=JACC.LaunchSpec(threads=512,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*512,f,args...);nothing
end
launchinit(A,n,dim,seed,threshold)=jlaunch(initkernel,cld(n,512),A,n,dim,seed,threshold)
launchdist(A,B,C,h,w,pw,weight,E)=jlaunch(distancekernel,cld(h,16),A,B,C,h,w,pw,weight,E)
function launchupdate(col,args...)
 n=col ? args[5] : args[4];blocks=cld(min(112,n),16)
 jlaunch(col ? colkernel : rowkernel,blocks,args...);nothing
end
include("../study_common.jl")
main()
