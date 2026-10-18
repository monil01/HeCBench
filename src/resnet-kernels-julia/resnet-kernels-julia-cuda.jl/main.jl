using CUDA
include(joinpath(@__DIR__,"..","convolution_math.jl"))
function bt_kernel!(t,a,c)
 x=Int32(blockIdx().x-1);y=Int32(blockIdx().y-1);part=Int32(blockIdx().z-1);tx=Int32(threadIdx().x-1);ty=Int32(threadIdx().y-1)
 s=CuDynamicSharedArray(Float32,4608)
 st_bt_load!(s,a,x,y,0,tx,ty,part,c);sync_threads()
 v=st_bt_first(s,tx,ty);sync_threads();st_bt_store!(s,v,tx,ty);sync_threads()
 v2=st_bt_second(s,tx,ty);sync_threads();st_bt_output!(t,v2,x,y,part,tx,ty,c)
 return
end
function outer_kernel!(o,a,w,c)
 plane=Int32(blockIdx().x-1);part=Int32(blockIdx().y-1);tx=Int32(threadIdx().x-1);ty=Int32(threadIdx().y-1)
 s=CuDynamicSharedArray(Float32,8*c+32*c)
 st_outer_load!(s,a,plane,part,tx,ty,c);v=0f0;v2=0f0
 for k in Int32(0):c÷32-1
  st_outer_weight!(s,w,plane,k,tx,ty,c);sync_threads()
  v+=st_outer_chunk(s,k,tx,ty,c,false)
  if c==256;v2+=st_outer_chunk(s,k,tx,ty,c,true);end
  sync_threads()
 end
 st_outer_output!(o,v,v2,plane,part,tx,ty,c)
 return
end
function at_kernel!(o,ip,bias,scale,c)
 x=Int32(blockIdx().x-1);y=Int32(blockIdx().y-1);z=Int32(blockIdx().z-1);tx=Int32(threadIdx().x-1);ty=Int32(threadIdx().y-1)
 s=CuDynamicSharedArray(Float32,36);st_at_load!(s,ip,x,y,z,tx,ty,c);sync_threads()
 v=st_at_first(s,tx,ty);sync_threads();s[tx*6+ty+1]=v;sync_threads()
 st_at_output!(o,s,bias,scale,x,y,z,tx,ty,c)
 return
end
function one_kernel!(o,a,w,bias,scale,ci,co,oc,chunk,relu)
 tile=Int32(blockIdx().x-1);part=Int32(blockIdx().y-1);tx=Int32(threadIdx().x-1);ty=Int32(threadIdx().y-1)
 s=CuDynamicSharedArray(Float32,4*ci+chunk*oc+6*oc)
 st_one_load!(s,a,bias,scale,tile,part,tx,ty,ci,co,oc);sync_threads()
 for k in Int32(0):chunk:ci-1
  st_one_weight!(s,w,k,part,tx,ty,ci,co,oc,chunk);sync_threads()
  st_one_acc!(s,k,tx,ty,ci,oc,chunk);sync_threads()
 end
 st_one_output!(o,s,tile,part,tx,ty,ci,co,oc,chunk,relu)
 return
end
function launch!(f)
 a,w,b,s,t,ip,o,ci,co,mode=f
 if mode<2
  @cuda threads=(128,6) blocks=(4,4,Int(ci)÷128) shmem=18432 bt_kernel!(t,a,ci)
  @cuda threads=(Int(ci),ci==128 ? 8 : 4) blocks=(36,2) shmem=((8*ci+32*ci+8*ci)*4) outer_kernel!(ip,t,w,ci)
  @cuda threads=(6,6) blocks=(4,4,Int(ci)) shmem=144 at_kernel!(o,ip,b,s,ci)
 else
  oc=Int32(min(ci,co));chunk=Int32(ci==1024 ? 16 : ci==256 ? 32 : 64)
  @cuda threads=(Int(oc),4) blocks=(49,Int(co÷oc)) shmem=(4*ci+chunk*oc+6*oc)*4 one_kernel!(o,a,w,b,s,ci,co,oc,chunk,mode==2||mode==4)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
abspath(PROGRAM_FILE)==(@__FILE__) && study_main("cuda_jl",launch!)
