using CUDA
include(joinpath(@__DIR__,"..","study_constant.jl"))
function cmpfhd!(rmu,imu,rfhd,ifhd,x,y,z,samples::Int32,voxels::Int32)
 n=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 if n<samples
  i=n+Int32(1);xn=x[i];yn=y[i];zn=z[i];r=rfhd[i];im=ifhd[i]
  for m in Int32(0):voxels-Int32(1)
   base=Int32(3)*m;e=Float32(2pi)*(load_k_component(base)*xn+load_k_component(base+Int32(1))*yn+load_k_component(base+Int32(2))*zn);c=Base.FastMath.cos_fast(e);s=Base.FastMath.sin_fast(e);rm=rmu[m+1];imm=imu[m+1];r+=rm*c-imm*s;im+=imm*c+rm*s
  end
  rfhd[i]=r;ifhd[i]=im
 end
 return
end
function factory(rmu,imu,rout,iout,x,y,z,ns,chunkcounts)
 compiled=CUDA.@cuda launch=false cmpfhd!(rmu[1],imu[1],rout,iout,x,y,z,Int32(ns),Int32(chunkcounts[1]));symbol=CUDA.CuGlobal{NTuple{12288,Float32}}(compiled.fun.mod,"fhd_study_k")
 println("FHD_PTX_BEGIN");CUDA.code_ptx(stdout,cmpfhd!,Tuple{typeof(cudaconvert(rmu[1])),typeof(cudaconvert(imu[1])),typeof(cudaconvert(rout)),typeof(cudaconvert(iout)),typeof(cudaconvert(x)),typeof(cudaconvert(y)),typeof(cudaconvert(z)),Int32,Int32};kernel=true,dump_module=true);println("FHD_PTX_END")
 symbol,(j)->compiled(rmu[j],imu[j],rout,iout,x,y,z,Int32(ns),Int32(chunkcounts[j]);threads=256,blocks=cld(ns,256))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
CUDA.stream!(CUDA.legacy_stream())
study((T,n)->CUDA.zeros(T,n),factory)
