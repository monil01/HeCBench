using CUDA
function pathfinder_source!(wall,src,dst,debug,it,border,cols,t)
 tx=Int32(threadIdx().x-1);bx=Int32(blockIdx().x-1)
 shared=CuStaticSharedArray(Int32,500)
 small=Int32(250)-2it;start=small*bx-border;stop=start+249;x=start+tx
 validmin=max(-start,Int32(0));validmax=min(Int32(249),Int32(249)-(stop-cols+1))
 w=max(tx-1,validmin);e=min(tx+1,validmax);valid=validmin<=tx<=validmax
 if 0<=x<cols
  @inbounds shared[tx+1]=src[x+2]
 end
 CUDA.sync_threads()
 computed=false
 for i in Int32(0):it-1
  computed=false
  if i+1<=tx<=248-i && valid
   computed=true
   @inbounds shared[tx+251]=min(shared[w+1],shared[tx+1],shared[e+1])+wall[cols*(t+i)+x+2]
   if tx==11 && i==0
    @inbounds debug[src[x+2]+2]=Int32(1)
   end
  end
  CUDA.sync_threads()
  if i==it-1
   break
  end
  if computed
   @inbounds shared[tx+1]=shared[tx+251]
  end
  CUDA.sync_threads()
 end
 if computed
  @inbounds dst[x+2]=shared[tx+251]
 end
 nothing
end
allocate(v)=CuArray(v)
function launch(w,s,d,dbg,it,h,c,t,template)
 @cuda threads=250 blocks=c.rows*c.cols÷250 pathfinder_source!(w,s,d,dbg,Int32(it),Int32(h),Int32(c.cols),Int32(t))
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
