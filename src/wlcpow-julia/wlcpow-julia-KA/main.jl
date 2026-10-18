using CUDA,KernelAbstractions
include("../wlcpow_device.jl")
@kernel unsafe_indices=true function force_kernel!(force,co,vel,nb,bonds,l0,params,period,N,nt,grid,::Val{CAP})where CAP
 sh=@localmem Float32 (CAP,);tid=@index(Local,Linear);bid=@index(Group,Linear);sz=nt+Int32(1)
 for i in tid:128:sz
  @inbounds for t in 1:8;sh[(t-1)*sz+i]=params[t][i];end
 end
 @synchronize
 i=(bid-1)*128+tid;stride=grid*128
 while i<=N;wsite!(i,force,co,vel,nb,bonds,l0,sh,period,N,nt);i+=stride;end
end
const kernel=force_kernel!(CUDABackend(),128)
function launch(c)
 kernel(map(a->a.data,c.force),c.co.data,c.vel.data,c.nb.data,c.bonds.data,c.l0.data,map(a->a.data,c.params),c.period,c.N,c.nt,Int32(c.blocks),Val(8*(Int(c.nt)+1));ndrange=c.blocks*128)
 nothing
end
include("../study_common.jl")
wlcpow_study(launch)
