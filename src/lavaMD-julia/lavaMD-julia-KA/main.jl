using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_types.jl"))
@kernel function md_kernel!(meta,rv,qv,fv,alpha::Float32,nb::Int32)
 BX=Int32(@index(Group,Linear))-Int32(1);TX=Int32(@index(Local,Linear))-Int32(1);sh=@localmem(Float32,(900,))

 if BX<nb
  a2=2f0*alpha*alpha;first_i=box_offset(meta,BX);wtx=TX
  while wtx<Int32(100)
   @inbounds r=rv[first_i+Int64(wtx)+1];off=4*wtx;sh[off+1]=r.v;sh[off+2]=r.x;sh[off+3]=r.y;sh[off+4]=r.z
   wtx+=Int32(128)
  end
  @synchronize
  for k in Int32(0):box_nn(meta,BX)
   pointer=k==0 ? BX : box_neighbor(meta,BX,k-Int32(1));first_j=box_offset(meta,pointer);wtx=TX
   while wtx<Int32(100)
    @inbounds r=rv[first_j+Int64(wtx)+1];off=Int32(400)+4*wtx;sh[off+1]=r.v;sh[off+2]=r.x;sh[off+3]=r.y;sh[off+4]=r.z;sh[Int32(801)+wtx]=qv[first_j+Int64(wtx)+1]
    wtx+=Int32(128)
   end
   @synchronize
   wtx=TX
   while wtx<Int32(100)
    off=4*wtx
    @inbounds av=sh[off+1];ax=sh[off+2];ay=sh[off+3];az=sh[off+4]
    for j in Int32(0):Int32(99)
     offb=Int32(400)+4*j
     @inbounds bv=sh[offb+1];bx=sh[offb+2];by=sh[offb+3];bz=sh[offb+4];q=sh[Int32(801)+j]
     r2=av+bv-(ax*bx+ay*by+az*bz);u2=a2*r2;vij=exp(-u2);fs=2f0*vij;dx=ax-bx;fx=fs*dx;dy=ay-by;fy=fs*dy;dz=az-bz;fz=fs*dz
     ii=first_i+Int64(wtx)+1;@inbounds old=fv[ii];fv[ii]=StudyFour(old.v+q*vij,old.x+q*fx,old.y+q*fy,old.z+q*fz)
    end
    wtx+=Int32(128)
   end
   @synchronize
  end
 end
end
function launch(meta,rv,q,fv,alpha,nb)
 md_kernel!(BACKEND,128)(meta,rv,q,fv,alpha,Int32(nb);ndrange=nb*128)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
