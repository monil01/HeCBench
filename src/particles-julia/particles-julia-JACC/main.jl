using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function integrate_k(index,pos,vel,p,dt,N)
 i=Int32(index)-Int32(1);if i<N;integrate_one(i,pos,vel,p,dt);end;return
end
function hash_k(index,keys,indices,pos,p,N)
 i=Int32(index)-Int32(1);if i<N;hash_one(i,keys,indices,pos,p);end;return
end
function clear_k(index,starts,C)
 i=Int32(index)-Int32(1);if i<C;@inbounds starts[i+1]=typemax(UInt32);end;return
end
function initial_sort_k(index,keys,indices,template)
 tid=Int32((index-1)%256);group=Int32((index-1)÷256);scratch=JACC.shared(template);load_sort!(tid,group,keys,indices,scratch);size=Int32(2)
 while size<Int32(512)
  dir=(tid&(size÷Int32(2)))!=0;stride=size÷Int32(2)
  while stride>0;JACC.sync_workgroup();a=Int32(2)*tid-(tid&(stride-Int32(1)));compare_shared!(scratch,a,a+stride,dir);stride>>=1;end;size<<=1
 end
 dir=(group&Int32(1))!=0;stride=Int32(256)
 while stride>0;JACC.sync_workgroup();a=Int32(2)*tid-(tid&(stride-Int32(1)));compare_shared!(scratch,a,a+stride,dir);stride>>=1;end
 JACC.sync_workgroup();store_sort!(tid,group,keys,indices,scratch);return
end
function merge_global_k(index,keys,indices,size,stride,N)
 i=Int32(index)-Int32(1);merge_global_one(i,keys,indices,size,stride,N);return
end
function merge_local_k(index,keys,indices,size,N,template)
 tid=Int32((index-1)%256);group=Int32((index-1)÷256);scratch=JACC.shared(template);load_sort!(tid,group,keys,indices,scratch);ci=(group*Int32(256)+tid)&(N÷Int32(2)-Int32(1));dir=(ci&(size÷Int32(2)))!=0;stride=Int32(256)
 while stride>0;JACC.sync_workgroup();a=Int32(2)*tid-(tid&(stride-Int32(1)));compare_shared!(scratch,a,a+stride,dir);stride>>=1;end
 JACC.sync_workgroup();store_sort!(tid,group,keys,indices,scratch);return
end
function reorder_k(index,pos,vel,rp,rv,keys,indices,starts,ends,N,template)
 tid=Int32((index-1)%64);i=Int32(index)-Int32(1);scratch=JACC.shared(template);h=reorder_pre(i,tid,keys,scratch,N);JACC.sync_workgroup();reorder_post(i,tid,h,keys,indices,pos,vel,rp,rv,starts,ends,scratch,N);return
end
function collide_k(index,vel,rp,rv,indices,starts,ends,p,N)
 i=Int32(index)-Int32(1);if i<N;collide_one(i,vel,rp,rv,indices,starts,ends,p);end;return
end
function factory(pos,vel,rp,rv,keys,indices,starts,ends,p,N,C,dt)
 spec64=JACC.launch_spec(;threads=64,blocks=256,shmem_size=0,sync=false);specClear=JACC.launch_spec(;threads=64,blocks=4096,shmem_size=0,sync=false);specSort=JACC.launch_spec(;threads=256,blocks=32,shmem_size=4096,sync=false);specGlobal=JACC.launch_spec(;threads=128,blocks=64,shmem_size=0,sync=false);specReorder=JACC.launch_spec(;threads=64,blocks=256,shmem_size=260,sync=false);sortTemplate=CUDA.zeros(UInt32,1024);reorderTemplate=CUDA.zeros(UInt32,65)
 ()->begin
  JACC.parallel_for(spec64,16384,integrate_k,pos,vel,p,dt,N);JACC.parallel_for(spec64,16384,hash_k,keys,indices,pos,p,N);JACC.parallel_for(specSort,8192,initial_sort_k,keys,indices,sortTemplate)
  size=Int32(1024)
  while size<=N
   stride=size÷Int32(2)
   while stride>0
    if stride>=Int32(512);JACC.parallel_for(specGlobal,8192,merge_global_k,keys,indices,size,stride,N)
    else;JACC.parallel_for(specSort,8192,merge_local_k,keys,indices,size,N,sortTemplate);break;end
    stride>>=1
   end
   size<<=1
  end
  JACC.parallel_for(specClear,262144,clear_k,starts,C);JACC.parallel_for(specReorder,16384,reorder_k,pos,vel,rp,rv,keys,indices,starts,ends,N,reorderTemplate);JACC.parallel_for(spec64,16384,collide_k,vel,rp,rv,indices,starts,ends,p,N)
 end
end
main()=particles_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
