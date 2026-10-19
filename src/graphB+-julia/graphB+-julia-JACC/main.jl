using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
include(joinpath(@__DIR__,"..","study_device.jl"))
function kernel1(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage1!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel2(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage2!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel3(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage3!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel4(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage4!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel5(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage5!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel6(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage6!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel7(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage7!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel8(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage8!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel9(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage9!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel10(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage10!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel11(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage11!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel12(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage12!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel13(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage13!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel14(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage14!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel15(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage15!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel16(i,d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(i-1);stage16!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function launch(stage,s;root=0,start=0,stop=0,level=0,seed=0)
 d=s.device;n=Int32(s.n);e=Int32(s.e);root=Int32(root);start=Int32(start);stop=Int32(stop);level=Int32(level);seed=Int32(seed);inc=Int32(s.blocks*256)
 spec=JACC.launch_spec(threads=256,blocks=s.blocks,shmem_size=0,sync=false)
 if stage==1;JACC.parallel_for(spec,s.blocks*256,kernel1,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==2;JACC.parallel_for(spec,s.blocks*256,kernel2,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==3;JACC.parallel_for(spec,s.blocks*256,kernel3,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==4;JACC.parallel_for(spec,s.blocks*256,kernel4,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==5;JACC.parallel_for(spec,s.blocks*256,kernel5,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==6;JACC.parallel_for(spec,s.blocks*256,kernel6,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==7;JACC.parallel_for(spec,s.blocks*256,kernel7,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==8;JACC.parallel_for(spec,s.blocks*256,kernel8,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==9;JACC.parallel_for(spec,s.blocks*256,kernel9,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==10;JACC.parallel_for(spec,s.blocks*256,kernel10,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==11;JACC.parallel_for(spec,s.blocks*256,kernel11,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==12;JACC.parallel_for(spec,s.blocks*256,kernel12,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==13;JACC.parallel_for(spec,s.blocks*256,kernel13,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==14;JACC.parallel_for(spec,s.blocks*256,kernel14,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==15;JACC.parallel_for(spec,s.blocks*256,kernel15,d,n,e,root,start,stop,level,seed,inc)
 elseif stage==16;JACC.parallel_for(spec,s.blocks*256,kernel16,d,n,e,root,start,stop,level,seed,inc)
 else;error("unknown stage");end;nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
