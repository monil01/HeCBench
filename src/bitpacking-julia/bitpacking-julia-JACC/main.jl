using CUDA
import JACC
JACC.@init_backend
@inline block_barrier()=JACC.sync_workgroup()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function scan!(i,lo,hi,input,num,tmp,grid)
 index=Int32(i)-Int32(1);s=JACC.shared(tmp);scan_body!(index%Int32(256),index÷Int32(256),grid,lo,hi,input,num,s);return
end
function finalize!(i,lo,hi,bits,minimum,num,tmp)
 s=JACC.shared(tmp);finalize_body!(Int32(i)-Int32(1),lo,hi,bits,minimum,num,s);return
end
function pack!(i,bits,minimum,out,input,num,tmp,grid)
 index=Int32(i)-Int32(1);s=JACC.shared(tmp);pack_body!(index%Int32(256),index÷Int32(256),grid,bits,minimum,out,input,num,s);return
end
function run!(input,num,lo,hi,bits,minimum,out,tmp,packtmp)
 grid=Int32(length(lo));extent=Int(grid)*256
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=Int(grid),shmem_size=2048,sync=false),extent,scan!,lo,hi,input,num,tmp,grid)
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=1,shmem_size=2048,sync=false),256,finalize!,lo,hi,bits,minimum,num,tmp)
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=Int(grid),shmem_size=1024,sync=false),extent,pack!,bits,minimum,out,input,num,packtmp,grid)
 return
end
study(run!)
