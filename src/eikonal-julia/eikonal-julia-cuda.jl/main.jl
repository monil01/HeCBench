using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function solver!(speed,mask,input,output,con,list,n::Int32,nIter::Int32,old::Int32)
 check=false
 sh=CuStaticSharedArray(Float64,216)
 li=Int32(blockIdx().x)-Int32(1);tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);tz=Int32(threadIdx().z)-Int32(1)
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
  sync_threads()
  oldT=sh[smidx(sx,sy,sz)];newT=oldT
  for iter in Int32(1):nIter
   oldT=sh[smidx(sx,sy,sz)];newT=oldT
   if valid
    a=min(sh[smidx(tx,sy,sz)],sh[smidx(tx+Int32(2),sy,sz)]);b=min(sh[smidx(sx,ty,sz)],sh[smidx(sx,ty+Int32(2),sz)]);c=min(sh[smidx(sx,sy,tz)],sh[smidx(sx,sy,tz+Int32(2))]);newT=min(source_time(a,b,c,F),oldT)
   end
   sync_threads()
   if valid;sh[smidx(sx,sy,sz)]=newT;end
   sync_threads()
  end
  con[idx]=(oldT-newT)<1e-16
  if !check||valid;output[idx]=newT;end
 end
 return
end
function neighbor!(speed,mask,input,output,con,list,n::Int32,nIter::Int32,old::Int32)
 check=true
 sh=CuStaticSharedArray(Float64,216)
 li=Int32(blockIdx().x)-Int32(1);tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);tz=Int32(threadIdx().z)-Int32(1)
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
  sync_threads()
  oldT=sh[smidx(sx,sy,sz)];newT=oldT
  for iter in Int32(1):nIter
   oldT=sh[smidx(sx,sy,sz)];newT=oldT
   if valid
    a=min(sh[smidx(tx,sy,sz)],sh[smidx(tx+Int32(2),sy,sz)]);b=min(sh[smidx(sx,ty,sz)],sh[smidx(sx,ty+Int32(2),sz)]);c=min(sh[smidx(sx,sy,tz)],sh[smidx(sx,sy,tz+Int32(2))]);newT=min(source_time(a,b,c,F),oldT)
   end
   sync_threads()
   if valid;sh[smidx(sx,sy,sz)]=newT;end
   sync_threads()
  end
  con[idx]=(oldT-newT)<1e-16
  if !check||valid;output[idx]=newT;end
 end
 return
end
function reduce!(con,vol,list)
 sh=CuStaticSharedArray(Bool,64)
 li=Int32(blockIdx().x)-Int32(1);tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);tz=Int32(threadIdx().z)-Int32(1);tid=tz*Int32(16)+ty*Int32(4)+tx;bi=Int32(list[li+1]);base=bi*Int32(64)
 sh[tid+1]=con[base+tid+1];sh[tid+33]=con[base+tid+33];sync_threads()
 stride=Int32(32)
 while stride>0
  if tid<stride;sh[tid+1]=sh[tid+1]&&sh[tid+stride+1];end
  sync_threads();stride÷=Int32(2)
 end
 if tid==0;vol[bi+1]=!sh[1];end
 return
end
function launch_stage(check,speed,mask,input,output,con,list,n,nIter,old,total)
 if check
  @cuda threads=(4,4,4) blocks=total neighbor!(speed,mask,input,output,con,list,Int32(n),Int32(nIter),Int32(old))
 else
  @cuda threads=(4,4,4) blocks=total solver!(speed,mask,input,output,con,list,Int32(n),Int32(nIter),Int32(old))
 end
end
launch_reduce(con,vol,list,total)=(@cuda threads=(4,4,2) blocks=total reduce!(con,vol,list))
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch_stage,launch_reduce)
