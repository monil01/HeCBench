using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function integrate_k(pos,vel,p,dt,N)
 i=Int32(@index(Global,Linear))-Int32(1);if i<N;integrate_one(i,pos,vel,p,dt);end
end
@kernel function hash_k(keys,indices,pos,p,N)
 i=Int32(@index(Global,Linear))-Int32(1);if i<N;hash_one(i,keys,indices,pos,p);end
end
@kernel function clear_k(starts,C)
 i=Int32(@index(Global,Linear))-Int32(1);if i<C;@inbounds starts[i+1]=typemax(UInt32);end
end
@kernel function initial_sort_k(keys,indices)
 tid=Int32(@index(Local,Linear))-Int32(1);group=Int32(@index(Group,Linear))-Int32(1);scratch=@localmem UInt32 (1024,);load_sort!(tid,group,keys,indices,scratch);size=Int32(2)
 while size<Int32(512)
  dir=(tid&(size÷Int32(2)))!=0;stride=size÷Int32(2)
  while stride>0;@synchronize;a=Int32(2)*tid-(tid&(stride-Int32(1)));compare_shared!(scratch,a,a+stride,dir);stride>>=1;end;size<<=1
 end
 dir=(group&Int32(1))!=0;stride=Int32(256)
 while stride>0;@synchronize;a=Int32(2)*tid-(tid&(stride-Int32(1)));compare_shared!(scratch,a,a+stride,dir);stride>>=1;end
 @synchronize;store_sort!(tid,group,keys,indices,scratch)
end
@kernel function merge_global_k(keys,indices,size,stride,N)
 i=Int32(@index(Global,Linear))-Int32(1);merge_global_one(i,keys,indices,size,stride,N)
end
@kernel function merge_local_k(keys,indices,size,N)
 tid=Int32(@index(Local,Linear))-Int32(1);group=Int32(@index(Group,Linear))-Int32(1);scratch=@localmem UInt32 (1024,);load_sort!(tid,group,keys,indices,scratch);ci=(group*Int32(256)+tid)&(N÷Int32(2)-Int32(1));dir=(ci&(size÷Int32(2)))!=0;stride=Int32(256)
 while stride>0;@synchronize;a=Int32(2)*tid-(tid&(stride-Int32(1)));compare_shared!(scratch,a,a+stride,dir);stride>>=1;end
 @synchronize;store_sort!(tid,group,keys,indices,scratch)
end
@kernel function reorder_k(pos,vel,rp,rv,keys,indices,starts,ends,N)
 tid=Int32(@index(Local,Linear))-Int32(1);i=(Int32(@index(Group,Linear))-Int32(1))*blockDim().x+tid;scratch=@localmem UInt32 (65,);h=reorder_pre(i,tid,keys,scratch,N);@synchronize;reorder_post(i,tid,h,keys,indices,pos,vel,rp,rv,starts,ends,scratch,N)
end
@kernel function collide_k(vel,rp,rv,indices,starts,ends,p,N)
 i=Int32(@index(Global,Linear))-Int32(1);if i<N;collide_one(i,vel,rp,rv,indices,starts,ends,p);end
end
function factory(pos,vel,rp,rv,keys,indices,starts,ends,p,N,C,dt)
 backend=CUDABackend();integ=integrate_k(backend,64);hashlaunch=hash_k(backend,64);initial=initial_sort_k(backend,256);globalmerge=merge_global_k(backend,128);localmerge=merge_local_k(backend,256);clearlaunch=clear_k(backend,64);reorderlaunch=reorder_k(backend,64);collision=collide_k(backend,64)
 ()->begin
  integ(pos,vel,p,dt,N;ndrange=16384);hashlaunch(keys,indices,pos,p,N;ndrange=16384);initial(keys,indices;ndrange=8192)
  size=Int32(1024)
  while size<=N
   stride=size÷Int32(2)
   while stride>0
    if stride>=Int32(512);globalmerge(keys,indices,size,stride,N;ndrange=8192)
    else;localmerge(keys,indices,size,N;ndrange=8192);break;end
    stride>>=1
   end
   size<<=1
  end
  clearlaunch(starts,C;ndrange=262144);reorderlaunch(pos,vel,rp,rv,keys,indices,starts,ends,N;ndrange=16384);collision(vel,rp,rv,indices,starts,ends,p,N;ndrange=16384)
 end
end
main()=particles_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
