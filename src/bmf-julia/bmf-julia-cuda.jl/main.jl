using CUDA
include("../study_device.jl")
function initkernel(A,n,dim,seed,threshold)
 initbody(A,n,dim,seed,threshold,Int32(CUDA.threadIdx().x)-1,Int32(CUDA.blockIdx().x)-1);return
end
function distancekernel(A,B,C,h,w,pw,weight,E)
 distbody(A,B,C,h,w,pw,weight,E,Int32(CUDA.threadIdx().x)-1,Int32(CUDA.blockIdx().x)-1);return
end
function rowkernel(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight)
 updatebody(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight,Int32(CUDA.threadIdx().x)-1,Int32(CUDA.blockIdx().x)-1,Val(false));return
end
function colkernel(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight)
 updatebody(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight,Int32(CUDA.threadIdx().x)-1,Int32(CUDA.blockIdx().x)-1,Val(true));return
end
launchinit(A,n,dim,seed,threshold)=CUDA.@cuda threads=512 blocks=cld(n,512) initkernel(A,n,dim,seed,threshold)
launchdist(A,B,C,h,w,pw,weight,E)=CUDA.@cuda threads=512 blocks=cld(h,16) distancekernel(A,B,C,h,w,pw,weight,E)
function launchupdate(col,args...)
 n=col ? args[5] : args[4];blocks=cld(min(112,n),16)
 if col;CUDA.@cuda threads=512 blocks=blocks colkernel(args...)
 else;CUDA.@cuda threads=512 blocks=blocks rowkernel(args...);end
 nothing
end
include("../study_common.jl")
main()
