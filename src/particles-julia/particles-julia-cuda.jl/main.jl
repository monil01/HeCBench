using CUDA
include("../study_common.jl")
function integrate_k(pos,vel,p,dt,N)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);if i<N;integrate_one(i,pos,vel,p,dt);end;return
end
function hash_k(keys,indices,pos,p,N)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);if i<N;hash_one(i,keys,indices,pos,p);end;return
end
function clear_k(starts,C)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);if i<C;@inbounds starts[i+1]=typemax(UInt32);end;return
end
function initial_sort_k(keys,indices)
 tid=threadIdx().x-Int32(1);group=blockIdx().x-Int32(1);scratch=CuStaticSharedArray(UInt32,1024);load_sort!(tid,group,keys,indices,scratch);size=Int32(2)
 while size<Int32(512)
  dir=(tid&(size÷Int32(2)))!=0;stride=size÷Int32(2)
  while stride>0;sync_threads();a=Int32(2)*tid-(tid&(stride-Int32(1)));compare_shared!(scratch,a,a+stride,dir);stride>>=1;end;size<<=1
 end
 dir=(group&Int32(1))!=0;stride=Int32(256)
 while stride>0;sync_threads();a=Int32(2)*tid-(tid&(stride-Int32(1)));compare_shared!(scratch,a,a+stride,dir);stride>>=1;end
 sync_threads();store_sort!(tid,group,keys,indices,scratch);return
end
function merge_global_k(keys,indices,size,stride,N)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);merge_global_one(i,keys,indices,size,stride,N);return
end
function merge_local_k(keys,indices,size,N)
 tid=threadIdx().x-Int32(1);group=blockIdx().x-Int32(1);scratch=CuStaticSharedArray(UInt32,1024);load_sort!(tid,group,keys,indices,scratch);ci=(group*Int32(256)+tid)&(N÷Int32(2)-Int32(1));dir=(ci&(size÷Int32(2)))!=0;stride=Int32(256)
 while stride>0;sync_threads();a=Int32(2)*tid-(tid&(stride-Int32(1)));compare_shared!(scratch,a,a+stride,dir);stride>>=1;end
 sync_threads();store_sort!(tid,group,keys,indices,scratch);return
end
function reorder_k(pos,vel,rp,rv,keys,indices,starts,ends,N)
 tid=threadIdx().x-Int32(1);i=(blockIdx().x-Int32(1))*blockDim().x+tid;scratch=CuStaticSharedArray(UInt32,65);h=reorder_pre(i,tid,keys,scratch,N);sync_threads();reorder_post(i,tid,h,keys,indices,pos,vel,rp,rv,starts,ends,scratch,N);return
end
function collide_k(vel,rp,rv,indices,starts,ends,p,N)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);if i<N;collide_one(i,vel,rp,rv,indices,starts,ends,p);end;return
end
function factory(pos,vel,rp,rv,keys,indices,starts,ends,p,N,C,dt)
 ()->begin
  @cuda threads=64 blocks=256 integrate_k(pos,vel,p,dt,N)
  @cuda threads=64 blocks=256 hash_k(keys,indices,pos,p,N)
  @cuda threads=256 blocks=32 initial_sort_k(keys,indices)
  size=Int32(1024)
  while size<=N
   stride=size÷Int32(2)
   while stride>0
    if stride>=Int32(512);@cuda threads=128 blocks=64 merge_global_k(keys,indices,size,stride,N)
    else;@cuda threads=256 blocks=32 merge_local_k(keys,indices,size,N);break;end
    stride>>=1
   end
   size<<=1
  end
  @cuda threads=64 blocks=4096 clear_k(starts,C)
  @cuda threads=64 blocks=256 reorder_k(pos,vel,rp,rv,keys,indices,starts,ends,N)
  @cuda threads=64 blocks=256 collide_k(vel,rp,rv,indices,starts,ends,p,N)
 end
end
main()=particles_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
