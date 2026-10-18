using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function solver!(speed,mask,input,output,con,list,n::Int32,nIter::Int32,old::Int32)
 check=false
 sh=@localmem(Float64,(216,))
 li=Int32(@index(Group,Linear))-Int32(1);tx=Int32(@index(Local,Cartesian)[1])-Int32(1);ty=Int32(@index(Local,Cartesian)[2])-Int32(1);tz=Int32(@index(Local,Cartesian)[3])-Int32(1)
 bi=Int32(list[li+1]);tid=tz*Int32(16)+ty*Int32(4)+tx;idx=bi*Int32(64)+tid+Int32(1)
 if check&&li<old
  output[idx]=input[idx]
 else
  nx=n÷Int32(4);bx=bi%nx;by=(bi÷nx)%nx;bz=bi÷(nx*nx);x=bx*Int32(4)+tx;y=by*Int32(4)+ty;z=bz*Int32(4)+tz;sx=tx+Int32(1);sy=ty+Int32(1);sz=tz+Int32(1)
  sh[smidx(sx,sy,sz)]=input[idx];F=speed[idx];if F>0;F=1.0/F;end;valid=mask[idx]
  if tx==0;sh[smidx(0,sy,sz)]=input[globalidx(x-Int32(1),y,z,n)];end
  if tx==3;sh[smidx(5,sy,sz)]=input[globalidx(x+Int32(1),y,z,n)];end
  if ty==0;sh[smidx(sx,0,sz)]=input[globalidx(x,y-Int32(1),z,n)];end
  if ty==3;sh[smidx(sx,5,sz)]=input[globalidx(x,y+Int32(1),z,n)];end
  if tz==0;sh[smidx(sx,sy,0)]=input[globalidx(x,y,z-Int32(1),n)];end
  if tz==3;sh[smidx(sx,sy,5)]=input[globalidx(x,y,z+Int32(1),n)];end
  @synchronize
  oldT=sh[smidx(sx,sy,sz)];newT=oldT
  for iter in Int32(1):nIter
   oldT=sh[smidx(sx,sy,sz)];newT=oldT
   if valid
    a=min(sh[smidx(tx,sy,sz)],sh[smidx(tx+Int32(2),sy,sz)]);b=min(sh[smidx(sx,ty,sz)],sh[smidx(sx,ty+Int32(2),sz)]);c=min(sh[smidx(sx,sy,tz)],sh[smidx(sx,sy,tz+Int32(2))]);newT=min(source_time(a,b,c,F),oldT)
   end
   @synchronize
   if valid;sh[smidx(sx,sy,sz)]=newT;end
   @synchronize
  end
  con[idx]=(oldT-newT)<1e-16
  if !check||valid;output[idx]=newT;end
 end
end
@kernel function neighbor!(speed,mask,input,output,con,list,n::Int32,nIter::Int32,old::Int32)
 check=true
 sh=@localmem(Float64,(216,))
 li=Int32(@index(Group,Linear))-Int32(1);tx=Int32(@index(Local,Cartesian)[1])-Int32(1);ty=Int32(@index(Local,Cartesian)[2])-Int32(1);tz=Int32(@index(Local,Cartesian)[3])-Int32(1)
 bi=Int32(list[li+1]);tid=tz*Int32(16)+ty*Int32(4)+tx;idx=bi*Int32(64)+tid+Int32(1)
 if check&&li<old
  output[idx]=input[idx]
 else
  nx=n÷Int32(4);bx=bi%nx;by=(bi÷nx)%nx;bz=bi÷(nx*nx);x=bx*Int32(4)+tx;y=by*Int32(4)+ty;z=bz*Int32(4)+tz;sx=tx+Int32(1);sy=ty+Int32(1);sz=tz+Int32(1)
  sh[smidx(sx,sy,sz)]=input[idx];F=speed[idx];if F>0;F=1.0/F;end;valid=mask[idx]
  if tx==0;sh[smidx(0,sy,sz)]=input[globalidx(x-Int32(1),y,z,n)];end
  if tx==3;sh[smidx(5,sy,sz)]=input[globalidx(x+Int32(1),y,z,n)];end
  if ty==0;sh[smidx(sx,0,sz)]=input[globalidx(x,y-Int32(1),z,n)];end
  if ty==3;sh[smidx(sx,5,sz)]=input[globalidx(x,y+Int32(1),z,n)];end
  if tz==0;sh[smidx(sx,sy,0)]=input[globalidx(x,y,z-Int32(1),n)];end
  if tz==3;sh[smidx(sx,sy,5)]=input[globalidx(x,y,z+Int32(1),n)];end
  @synchronize
  oldT=sh[smidx(sx,sy,sz)];newT=oldT
  for iter in Int32(1):nIter
   oldT=sh[smidx(sx,sy,sz)];newT=oldT
   if valid
    a=min(sh[smidx(tx,sy,sz)],sh[smidx(tx+Int32(2),sy,sz)]);b=min(sh[smidx(sx,ty,sz)],sh[smidx(sx,ty+Int32(2),sz)]);c=min(sh[smidx(sx,sy,tz)],sh[smidx(sx,sy,tz+Int32(2))]);newT=min(source_time(a,b,c,F),oldT)
   end
   @synchronize
   if valid;sh[smidx(sx,sy,sz)]=newT;end
   @synchronize
  end
  con[idx]=(oldT-newT)<1e-16
  if !check||valid;output[idx]=newT;end
 end
end
@kernel function reduce!(con,vol,list)
 sh=@localmem(Bool,(64,))
 li=Int32(@index(Group,Linear))-Int32(1);tx=Int32(@index(Local,Cartesian)[1])-Int32(1);ty=Int32(@index(Local,Cartesian)[2])-Int32(1);tz=Int32(@index(Local,Cartesian)[3])-Int32(1);tid=tz*Int32(16)+ty*Int32(4)+tx;bi=Int32(list[li+1]);base=bi*Int32(64)
 sh[tid+1]=con[base+tid+1];sh[tid+33]=con[base+tid+33];@synchronize
 stride=Int32(32)
 while stride>0
  if tid<stride;sh[tid+1]=sh[tid+1]&&sh[tid+stride+1];end
  @synchronize;stride÷=Int32(2)
 end
 if tid==0;vol[bi+1]=!sh[1];end
end
function launch_stage(check,speed,mask,input,output,con,list,n,nIter,old,total)
 if check
  neighbor!(BACKEND,(4,4,4))(speed,mask,input,output,con,list,Int32(n),Int32(nIter),Int32(old);ndrange=(4total,4,4))
 else
  solver!(BACKEND,(4,4,4))(speed,mask,input,output,con,list,Int32(n),Int32(nIter),Int32(old);ndrange=(4total,4,4))
 end
end
launch_reduce(con,vol,list,total)=reduce!(BACKEND,(4,4,2))(con,vol,list;ndrange=(4total,4,2))
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch_stage,launch_reduce)
