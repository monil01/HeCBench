using CUDA
import JACC
JACC.@init_backend
const SOLVER_TEMPLATE=JACC.zeros(Float64,216)
const REDUCE_TEMPLATE=JACC.zeros(Bool,64)
include(joinpath(@__DIR__,"..","study_math.jl"))
function solver!(xx,yy,zz,speed,mask,input,output,con,list,n::Int32,nIter::Int32,old::Int32,template)
 check=false
 sh=JACC.shared(template)
 li=Int32(div(xx-1,4));tx=Int32(mod(xx-1,4));ty=Int32(mod(yy-1,4));tz=Int32(mod(zz-1,4))
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
  JACC.sync_workgroup()
  oldT=sh[smidx(sx,sy,sz)];newT=oldT
  for iter in Int32(1):nIter
   oldT=sh[smidx(sx,sy,sz)];newT=oldT
   if valid
    a=min(sh[smidx(tx,sy,sz)],sh[smidx(tx+Int32(2),sy,sz)]);b=min(sh[smidx(sx,ty,sz)],sh[smidx(sx,ty+Int32(2),sz)]);c=min(sh[smidx(sx,sy,tz)],sh[smidx(sx,sy,tz+Int32(2))]);newT=min(source_time(a,b,c,F),oldT)
   end
   JACC.sync_workgroup()
   if valid;sh[smidx(sx,sy,sz)]=newT;end
   JACC.sync_workgroup()
  end
  con[idx]=(oldT-newT)<1e-16
  if !check||valid;output[idx]=newT;end
 end
 return
end
function neighbor!(xx,yy,zz,speed,mask,input,output,con,list,n::Int32,nIter::Int32,old::Int32,template)
 check=true
 sh=JACC.shared(template)
 li=Int32(div(xx-1,4));tx=Int32(mod(xx-1,4));ty=Int32(mod(yy-1,4));tz=Int32(mod(zz-1,4))
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
  JACC.sync_workgroup()
  oldT=sh[smidx(sx,sy,sz)];newT=oldT
  for iter in Int32(1):nIter
   oldT=sh[smidx(sx,sy,sz)];newT=oldT
   if valid
    a=min(sh[smidx(tx,sy,sz)],sh[smidx(tx+Int32(2),sy,sz)]);b=min(sh[smidx(sx,ty,sz)],sh[smidx(sx,ty+Int32(2),sz)]);c=min(sh[smidx(sx,sy,tz)],sh[smidx(sx,sy,tz+Int32(2))]);newT=min(source_time(a,b,c,F),oldT)
   end
   JACC.sync_workgroup()
   if valid;sh[smidx(sx,sy,sz)]=newT;end
   JACC.sync_workgroup()
  end
  con[idx]=(oldT-newT)<1e-16
  if !check||valid;output[idx]=newT;end
 end
 return
end
function reduce!(xx,yy,zz,con,vol,list,template)
 sh=JACC.shared(template)
 li=Int32(div(xx-1,4));tx=Int32(mod(xx-1,4));ty=Int32(mod(yy-1,4));tz=Int32(mod(zz-1,4));tid=tz*Int32(16)+ty*Int32(4)+tx;bi=Int32(list[li+1]);base=bi*Int32(64)
 sh[tid+1]=con[base+tid+1];sh[tid+33]=con[base+tid+33];JACC.sync_workgroup()
 stride=Int32(32)
 while stride>0
  if tid<stride;sh[tid+1]=sh[tid+1]&&sh[tid+stride+1];end
  JACC.sync_workgroup();stride÷=Int32(2)
 end
 if tid==0;vol[bi+1]=!sh[1];end
 return
end
function launch_stage(check,speed,mask,input,output,con,list,n,nIter,old,total)
 spec=JACC.launch_spec(threads=(4,4,4),blocks=(total,1,1),shmem_size=1728,sync=false)
 if check
  JACC.parallel_for(spec,(4total,4,4),neighbor!,speed,mask,input,output,con,list,Int32(n),Int32(nIter),Int32(old),SOLVER_TEMPLATE)
 else
  JACC.parallel_for(spec,(4total,4,4),solver!,speed,mask,input,output,con,list,Int32(n),Int32(nIter),Int32(old),SOLVER_TEMPLATE)
 end
end
function launch_reduce(con,vol,list,total)
 spec=JACC.launch_spec(threads=(4,4,2),blocks=(total,1,1),shmem_size=64,sync=false)
 JACC.parallel_for(spec,(4total,4,2),reduce!,con,vol,list,REDUCE_TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch_stage,launch_reduce)
