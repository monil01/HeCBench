using CUDA,KernelAbstractions
const MODEL="ka"
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function kernel1(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage1!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel2(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage2!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel3(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage3!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel4(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage4!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel5(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage5!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel6(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage6!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel7(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage7!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel8(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage8!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel9(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage9!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel10(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage10!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel11(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage11!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel12(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage12!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel13(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage13!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel14(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage14!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel15(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage15!(d,n,e,root,start,stop,level,seed,tid,inc)
end
@kernel function kernel16(d,n,e,root,start,stop,level,seed,inc)
 tid=Int32(@index(Global,Linear))-Int32(1)
 stage16!(d,n,e,root,start,stop,level,seed,tid,inc)
end
function launch(stage,s;root=0,start=0,stop=0,level=0,seed=0)
 d=s.device;n=Int32(s.n);e=Int32(s.e);root=Int32(root);start=Int32(start);stop=Int32(stop);level=Int32(level);seed=Int32(seed);inc=Int32(s.blocks*256)
 if stage==1;kernel1(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==2;kernel2(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==3;kernel3(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==4;kernel4(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==5;kernel5(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==6;kernel6(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==7;kernel7(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==8;kernel8(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==9;kernel9(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==10;kernel10(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==11;kernel11(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==12;kernel12(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==13;kernel13(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==14;kernel14(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==15;kernel15(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 elseif stage==16;kernel16(CUDABackend(),256)(d,n,e,root,start,stop,level,seed,inc;ndrange=s.blocks*256)
 else;error("unknown stage");end;nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
