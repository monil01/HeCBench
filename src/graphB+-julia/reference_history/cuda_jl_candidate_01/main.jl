using CUDA
const MODEL="cuda_jl"
include(joinpath(@__DIR__,"..","study_device.jl"))
function kernel1(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage1!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel2(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage2!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel3(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage3!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel4(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage4!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel5(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage5!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel6(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage6!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel7(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage7!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel8(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage8!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel9(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage9!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel10(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage10!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel11(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage11!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel12(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage12!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel13(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage13!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel14(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage14!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel15(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage15!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function kernel16(d,n,e,root,start,stop,level,seed)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(256)+CUDA.threadIdx().x-Int32(1);inc=CUDA.gridDim().x*Int32(256)
 stage16!(d,n,e,root,start,stop,level,seed,tid,inc);nothing
end
function launch(stage,s;root=0,start=0,stop=0,level=0,seed=0)
 d=s.device;n=Int32(s.n);e=Int32(s.e);root=Int32(root);start=Int32(start);stop=Int32(stop);level=Int32(level);seed=Int32(seed)
 if stage==1;CUDA.@cuda threads=256 blocks=s.blocks kernel1(d,n,e,root,start,stop,level,seed)
 elseif stage==2;CUDA.@cuda threads=256 blocks=s.blocks kernel2(d,n,e,root,start,stop,level,seed)
 elseif stage==3;CUDA.@cuda threads=256 blocks=s.blocks kernel3(d,n,e,root,start,stop,level,seed)
 elseif stage==4;CUDA.@cuda threads=256 blocks=s.blocks kernel4(d,n,e,root,start,stop,level,seed)
 elseif stage==5;CUDA.@cuda threads=256 blocks=s.blocks kernel5(d,n,e,root,start,stop,level,seed)
 elseif stage==6;CUDA.@cuda threads=256 blocks=s.blocks kernel6(d,n,e,root,start,stop,level,seed)
 elseif stage==7;CUDA.@cuda threads=256 blocks=s.blocks kernel7(d,n,e,root,start,stop,level,seed)
 elseif stage==8;CUDA.@cuda threads=256 blocks=s.blocks kernel8(d,n,e,root,start,stop,level,seed)
 elseif stage==9;CUDA.@cuda threads=256 blocks=s.blocks kernel9(d,n,e,root,start,stop,level,seed)
 elseif stage==10;CUDA.@cuda threads=256 blocks=s.blocks kernel10(d,n,e,root,start,stop,level,seed)
 elseif stage==11;CUDA.@cuda threads=256 blocks=s.blocks kernel11(d,n,e,root,start,stop,level,seed)
 elseif stage==12;CUDA.@cuda threads=256 blocks=s.blocks kernel12(d,n,e,root,start,stop,level,seed)
 elseif stage==13;CUDA.@cuda threads=256 blocks=s.blocks kernel13(d,n,e,root,start,stop,level,seed)
 elseif stage==14;CUDA.@cuda threads=256 blocks=s.blocks kernel14(d,n,e,root,start,stop,level,seed)
 elseif stage==15;CUDA.@cuda threads=256 blocks=s.blocks kernel15(d,n,e,root,start,stop,level,seed)
 elseif stage==16;CUDA.@cuda threads=256 blocks=s.blocks kernel16(d,n,e,root,start,stop,level,seed)
 else;error("unknown stage");end;nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
