using CUDA
import JACC
JACC.@init_backend
function pathfinder_jacc!(index,wall,src,dst,debug,it,border,cols,t,template)
 tx=Int32((index-1)%250);bx=Int32((index-1)÷250)
 shared=JACC.shared(template)
 small=Int32(250)-2it;start=small*bx-border;stop=start+249;x=start+tx
 validmin=max(-start,Int32(0));validmax=min(Int32(249),Int32(249)-(stop-cols+1))
 w=max(tx-1,validmin);e=min(tx+1,validmax);valid=validmin<=tx<=validmax
 if 0<=x<cols
  @inbounds shared[tx+1]=src[x+2]
 end
 JACC.sync_workgroup()
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
  JACC.sync_workgroup()
  if i==it-1
   break
  end
  if computed
   @inbounds shared[tx+1]=shared[tx+251]
  end
  JACC.sync_workgroup()
 end
 if computed
  @inbounds dst[x+2]=shared[tx+251]
 end
 nothing
end
allocate(v)=JACC.array(v)
function launch(w,s,d,dbg,it,h,c,t,template)
 blocks=c.rows*c.cols÷250
 spec=JACC.launch_spec(;threads=250,blocks=blocks,shmem_size=2000,sync=false)
 JACC.parallel_for(spec,blocks*250,pathfinder_jacc!,w,s,d,dbg,Int32(it),Int32(h),Int32(c.cols),Int32(t),template)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
