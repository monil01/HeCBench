using CUDA
import JACC
JACC.@init_backend
@inline block_barrier()=JACC.sync_workgroup()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function base!(i,j,s,N,f,w,h,d)
 base_body!(Int32(i)-Int32(1),Int32(j)-Int32(1),s,N,f,w,h,d);return
end
function tile!(i,j,s,N,f,w,h,d,tmp)
 x=Int32(i)-Int32(1);y=Int32(j)-Int32(1);shared=JACC.shared(tmp);tile_body!(x,y,(y%Int32(16))*Int32(16)+x%Int32(16),s,N,f,w,h,d,shared);return
end
function run!(tiled,s,N,f,w,h,d,tmp)
 if tiled;JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=(2,2),shmem_size=7168,sync=false),(32,32),tile!,s,N,f,w,h,d,tmp)
 else;JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=(2,2),shmem_size=0,sync=false),(32,32),base!,s,N,f,w,h,d)
 end
 return
end
study(run!)
