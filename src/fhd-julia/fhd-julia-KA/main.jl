using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_constant.jl"))
@kernel function cmpfhd!(rmu,imu,rfhd,ifhd,x,y,z,samples::Int32,voxels::Int32)
 n=Int32(@index(Global,Linear))-Int32(1)
 if n<samples
  i=n+Int32(1);xn=x[i];yn=y[i];zn=z[i];r=rfhd[i];im=ifhd[i]
  for m in Int32(0):voxels-Int32(1)
   base=Int32(3)*m;e=Float32(2pi)*(load_k_component(base)*xn+load_k_component(base+Int32(1))*yn+load_k_component(base+Int32(2))*zn);c=Base.FastMath.cos_fast(e);s=Base.FastMath.sin_fast(e);rm=rmu[m+1];imm=imu[m+1];r+=rm*c-imm*s;im+=imm*c+rm*s
  end
  rfhd[i]=r;ifhd[i]=im
 end
end
function factory(rmu,imu,rout,iout,x,y,z,ns,chunkcounts)
 backend=CUDA.CUDABackend();obj=cmpfhd!(backend,256);N=cld(ns,256)*256
 ndr,wgs,iterspace,dynamic=KernelAbstractions.launch_config(obj,N,nothing);ctx=KernelAbstractions.mkcontext(obj,ndr,iterspace)
 compiled=CUDA.@cuda launch=false always_inline=backend.always_inline maxthreads=256 obj.f(ctx,rmu[1],imu[1],rout,iout,x,y,z,Int32(ns),Int32(chunkcounts[1]))
 symbol=CUDA.CuGlobal{NTuple{12288,Float32}}(compiled.fun.mod,"fhd_study_k")
 args=(ctx,rmu[1],imu[1],rout,iout,x,y,z,Int32(ns),Int32(chunkcounts[1]));println("FHD_PTX_BEGIN");CUDA.code_ptx(stdout,obj.f,Tuple{map(a->typeof(cudaconvert(a)),args)...};kernel=true,dump_module=true,maxthreads=256,always_inline=backend.always_inline);println("FHD_PTX_END")
 symbol,(j)->obj(rmu[j],imu[j],rout,iout,x,y,z,Int32(ns),Int32(chunkcounts[j]);ndrange=N)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
CUDA.stream!(CUDA.legacy_stream())
study((T,n)->KernelAbstractions.allocate(CUDA.CUDABackend(),T,n),factory)
