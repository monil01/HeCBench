using CUDA,JACC
include("../wlcpow_device.jl")
function force_kernel!(index,force,co,vel,nb,bonds,l0,params,period,N,nt,grid,template)
 tid=rem(Int32(index)-Int32(1),Int32(128))+Int32(1);sh=JACC.shared(template);sz=nt+Int32(1)
 for i in tid:Int32(128):sz
  @inbounds for t in 1:8;sh[(t-1)*sz+i]=params[t][i];end
 end
 JACC.sync_workgroup();i=Int32(index);stride=grid*Int32(128)
 while i<=N;wsite!(i,force,co,vel,nb,bonds,l0,sh,period,N,nt);i+=stride;end
 nothing
end
const templates=Dict(nt=>CUDA.zeros(Float32,8*(nt+1)) for nt in [0,1,4,32,3])
function launch(c)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=128,blocks=c.blocks,shmem_size=8*(Int(c.nt)+1)*4,sync=false)
 JACC.parallel_for(force_kernel!,spec,c.blocks*128,map(a->a.data,c.force),c.co.data,c.vel.data,c.nb.data,c.bonds.data,c.l0.data,map(a->a.data,c.params),c.period,c.N,c.nt,Int32(c.blocks),templates[Int(c.nt)])
 nothing
end
include("../study_common.jl")
wlcpow_study(launch)
