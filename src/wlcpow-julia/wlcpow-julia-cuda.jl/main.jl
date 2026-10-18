using CUDA
include("../wlcpow_device.jl")
function force_kernel!(force,co,vel,nb,bonds,l0,params,period,N,nt)
 sh=@cuDynamicSharedMem(Float32,8*(nt+Int32(1)));tid=Int32(threadIdx().x);sz=nt+Int32(1)
 for i in tid:Int32(blockDim().x):sz
  @inbounds for t in 1:8;sh[(t-1)*sz+i]=params[t][i];end
 end
 sync_threads();i=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;stride=Int32(gridDim().x*blockDim().x)
 while i<=N;wsite!(i,force,co,vel,nb,bonds,l0,sh,period,N,nt);i+=stride;end
 return
end
function launch(c)
 @cuda threads=128 blocks=c.blocks shmem=8*(Int(c.nt)+1)*4 force_kernel!(map(a->a.data,c.force),c.co.data,c.vel.data,c.nb.data,c.bonds.data,c.l0.data,map(a->a.data,c.params),c.period,c.N,c.nt)
 nothing
end
include("../study_common.jl")
wlcpow_study(launch)
