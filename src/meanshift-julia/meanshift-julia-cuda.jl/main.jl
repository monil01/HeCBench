using CUDA
@inline sync_mean()=CUDA.sync_threads()
include(joinpath(@__DIR__,"..","study_device.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function mean_base(data,out)
 tid=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 if tid<10000;meanshift_body(tid,data,out,data,false);end;return
end
function mean_tiled(data,out)
 tid=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);scratch=CUDA.@cuStaticSharedMem(Float32,256);meanshift_body(tid,data,out,scratch,true);return
end
function factory(a,b,mode)
 kernel=mode=="base" ? mean_base : mean_tiled
 ()->begin
  x=a;y=b
  for iteration in 1:1000
   @cuda threads=64 blocks=157 kernel(x,y);CUDA.synchronize();x,y=y,x
  end
 end
end
if abspath(PROGRAM_FILE)==@__FILE__;meanshift_study(factory);end
