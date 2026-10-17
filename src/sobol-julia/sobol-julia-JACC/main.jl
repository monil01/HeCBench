using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function sobol_jacc!(i,j,dirs,out,n,d,gx,template)
 tx=mod(Int32(i)-Int32(1),Int32(128));bx=(Int32(i)-Int32(1))÷Int32(128);by=Int32(j)-Int32(1)
 tile=JACC.shared(template)
 if tx<Int32(32)
  @inbounds tile[tx+Int32(1)]=dirs[by*Int32(32)+tx+Int32(1)]
 end
 JACC.sync_workgroup()
 sobol_body!(tx,bx,by,gx,tile,out,n)
 nothing
end
function run!(dirs,out,n,d,gx,template)
 spec=JACC.launch_spec(threads=(128,1),blocks=(Int(gx),Int(d)),shmem_size=128,sync=false)
 JACC.parallel_for(spec,(Int(gx)*128,Int(d)),sobol_jacc!,dirs,out,n,d,gx,template)
 nothing
end
study(run!)
