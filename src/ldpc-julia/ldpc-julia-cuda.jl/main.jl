using CUDA
include(joinpath(@__DIR__,"..","ldpc_math.jl"))
function cnp_first!(llr,dt,R,count,compact)
 x=Int32(threadIdx().x-1);y=Int32(threadIdx().y-1);r=Int32(blockIdx().x-1);mc=Int32(blockIdx().y-1)
 s=cnp_prepare!(Val(true),x,y,r,mc,llr,R,count,compact,nothing)
 cnp_finish!(Val(true),r,dt,R,compact,nothing,s)
 return
end
function cnp_normal!(llr,dt,R,count,compact)
 x=Int32(threadIdx().x-1);y=Int32(threadIdx().y-1);r=Int32(blockIdx().x-1);mc=Int32(blockIdx().y-1);cache=CuDynamicSharedArray(Float32,1344)
 s=cnp_prepare!(Val(false),x,y,r,mc,llr,R,count,compact,cache)
 sync_threads()
 cnp_finish!(Val(false),r,dt,R,compact,cache,s)
 return
end
function vnp!(last,llr,dt,hd,count,compact)
 vnp_site!(last,Int32(threadIdx().x-1),Int32(threadIdx().y-1),Int32(blockIdx().x-1),Int32(blockIdx().y-1),llr,dt,hd,count,compact)
 return
end
function decode(c)
 for it in 1:10
  if it==1
   @cuda threads=(96,2) blocks=(12,c.mcw) cnp_first!(c.llr,c.dt,c.R,c.count1,c.comp1)
  else
   @cuda threads=(96,2) blocks=(12,c.mcw) shmem=5376 cnp_normal!(c.llr,c.dt,c.R,c.count1,c.comp1)
  end
  @cuda threads=(96,2) blocks=(24,c.mcw) vnp!(Val(it==10),c.llr,c.dt,c.hd,c.count2,c.comp2)
 end
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
ldpc_study(decode)
