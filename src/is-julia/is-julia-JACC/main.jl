using CUDA
import JACC
JACC.@init_backend
@inline block_barrier()=JACC.sync_workgroup()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function stage0!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(256);bid=gid÷Int32(256);s=();stage_body!(Val(0),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage1!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(1);bid=gid÷Int32(1);s=();stage_body!(Val(1),tid,gid,bid,Int32(1),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage2!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(256);bid=gid÷Int32(256);s=();stage_body!(Val(2),tid,gid,bid,Int32(8),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage3!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(256);bid=gid÷Int32(256);s=();stage_body!(Val(3),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage4!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(256);bid=gid÷Int32(256);s=JACC.shared(tmp);stage_body!(Val(4),tid,gid,bid,Int32(8),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage5!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(256);bid=gid÷Int32(256);s=JACC.shared(tmp);stage_body!(Val(5),tid,gid,bid,Int32(1),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage6!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(256);bid=gid÷Int32(256);s=();stage_body!(Val(6),tid,gid,bid,Int32(8),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage7!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(1);bid=gid÷Int32(1);s=();stage_body!(Val(7),tid,gid,bid,Int32(1),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage8!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(256);bid=gid÷Int32(256);s=();stage_body!(Val(8),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage9!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(256);bid=gid÷Int32(256);s=();stage_body!(Val(9),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage10!(i,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,tmp)
 gid=Int32(i)-Int32(1);tid=gid%Int32(256);bid=gid÷Int32(256);s=JACC.shared(tmp);stage_body!(Val(10),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function run!(stage,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,tmp,verifytmp)
 if stage==0;JACC.parallel_for(JACC.launch_spec(threads=256,blocks=256,shmem_size=0,sync=false),65536,stage0!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==1;JACC.parallel_for(JACC.launch_spec(threads=1,blocks=1,shmem_size=0,sync=false),1,stage1!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==2;JACC.parallel_for(JACC.launch_spec(threads=256,blocks=8,shmem_size=0,sync=false),2048,stage2!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==3;JACC.parallel_for(JACC.launch_spec(threads=256,blocks=256,shmem_size=0,sync=false),65536,stage3!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==4;JACC.parallel_for(JACC.launch_spec(threads=256,blocks=8,shmem_size=2048,sync=false),2048,stage4!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==5;JACC.parallel_for(JACC.launch_spec(threads=256,blocks=1,shmem_size=2048,sync=false),256,stage5!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==6;JACC.parallel_for(JACC.launch_spec(threads=256,blocks=8,shmem_size=0,sync=false),2048,stage6!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==7;JACC.parallel_for(JACC.launch_spec(threads=1,blocks=1,shmem_size=0,sync=false),1,stage7!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==8;JACC.parallel_for(JACC.launch_spec(threads=256,blocks=256,shmem_size=0,sync=false),65536,stage8!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==9;JACC.parallel_for(JACC.launch_spec(threads=256,blocks=256,shmem_size=0,sync=false),65536,stage9!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,tmp)
 elseif stage==10;JACC.parallel_for(JACC.launch_spec(threads=256,blocks=256,shmem_size=1024,sync=false),65536,stage10!,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0,verifytmp)
 end
 return
end
study(run!)
