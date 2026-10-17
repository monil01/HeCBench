using CUDA
@inline block_barrier()=CUDA.sync_threads()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function stage0!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=();stage_body!(Val(0),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage1!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=();stage_body!(Val(1),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage2!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=();stage_body!(Val(2),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage3!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=();stage_body!(Val(3),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage4!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=CuStaticSharedArray(Int32,512);stage_body!(Val(4),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage5!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=CuStaticSharedArray(Int32,512);stage_body!(Val(5),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage6!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=();stage_body!(Val(6),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage7!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=();stage_body!(Val(7),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage8!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=();stage_body!(Val(8),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage9!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=();stage_body!(Val(9),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function stage10!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+tid;bid=Int32(blockIdx().x)-Int32(1);s=CuStaticSharedArray(Int32,256);stage_body!(Val(10),tid,gid,bid,Int32(gridDim().x),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s);return
end
function run!(stage,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,tmp,verifytmp)
 if stage==0;@cuda threads=256 blocks=256 stage0!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==1;@cuda threads=1 blocks=1 stage1!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==2;@cuda threads=256 blocks=8 stage2!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==3;@cuda threads=256 blocks=256 stage3!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==4;@cuda threads=256 blocks=8 stage4!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==5;@cuda threads=256 blocks=1 stage5!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==6;@cuda threads=256 blocks=8 stage6!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==7;@cuda threads=1 blocks=1 stage7!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==8;@cuda threads=256 blocks=256 stage8!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==9;@cuda threads=256 blocks=256 stage9!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 elseif stage==10;@cuda threads=256 blocks=256 stage10!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0)
 end
 return
end
study(run!)
