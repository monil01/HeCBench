using CUDA, KernelAbstractions
@inline block_barrier()=KernelAbstractions.@synchronize()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel cpu=false function stage0!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=();stage_body!(Val(0),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage1!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=();stage_body!(Val(1),tid,gid,bid,Int32(1),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage2!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=();stage_body!(Val(2),tid,gid,bid,Int32(8),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage3!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=();stage_body!(Val(3),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage4!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=@localmem(Int32,(512,));stage_body!(Val(4),tid,gid,bid,Int32(8),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage5!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=@localmem(Int32,(512,));stage_body!(Val(5),tid,gid,bid,Int32(1),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage6!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=();stage_body!(Val(6),tid,gid,bid,Int32(8),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage7!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=();stage_body!(Val(7),tid,gid,bid,Int32(1),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage8!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=();stage_body!(Val(8),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage9!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=();stage_body!(Val(9),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
@kernel cpu=false function stage10!(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=Int32(@index(Global,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);s=@localmem(Int32,(256,));stage_body!(Val(10),tid,gid,bid,Int32(256),keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)
end
function run!(stage,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,tmp,verifytmp)
 if stage==0;stage0!(get_backend(keys),256)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=65536)
 elseif stage==1;stage1!(get_backend(keys),1)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=1)
 elseif stage==2;stage2!(get_backend(keys),256)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=2048)
 elseif stage==3;stage3!(get_backend(keys),256)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=65536)
 elseif stage==4;stage4!(get_backend(keys),256)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=2048)
 elseif stage==5;stage5!(get_backend(keys),256)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=256)
 elseif stage==6;stage6!(get_backend(keys),256)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=2048)
 elseif stage==7;stage7!(get_backend(keys),1)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=1)
 elseif stage==8;stage8!(get_backend(keys),256)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=65536)
 elseif stage==9;stage9!(get_backend(keys),256)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=65536)
 elseif stage==10;stage10!(get_backend(keys),256)(keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,314159265.0,1220703125.0;ndrange=65536)
 end
 return
end
study(run!)
