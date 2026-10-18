using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function mass!(x,y,node,R6,b,ny,zero,arrival,t)
    j=Int32(x-1)+b[2];i=Int32(y-1)+b[1]
    ew_stage1!(node,R6,b...,ny,zero,arrival,t,j,i)
    nothing
end
function momentum!(x,y,node,R6,b,ny)
    j=Int32(x-1)+b[2];i=Int32(y-1)+b[1]
    ew_stage3!(node,R6,b...,ny,j,i)
    nothing
end
function boundary_height!(index,node,C,b,ny,nx)
    ew_stage2!(node,C...,b...,ny,nx)
    nothing
end
function boundary_momentum!(index,node,C,B,ny,nx,clip)
    ew_stage4!(node,C...,B...,ny,nx,clip)
    nothing
end
function factory(node,R6,C,B,ny,nx)
 serial=JACC.launch_spec(;threads=1,blocks=1,shmem_size=0,sync=false)
 function launch(stage,b,t)
  range=(Int(b[4]-b[2]+1),Int(b[3]-b[1]+1))
  if stage==1 || stage==3
   spec=JACC.launch_spec(;threads=(16,16),blocks=(cld(range[1],16),cld(range[2],16)),shmem_size=0,sync=false)
   if stage==1
    JACC.parallel_for(spec,range,mass!,node,R6,b,ny,1f-6,1f-4,t)
   else
    JACC.parallel_for(spec,range,momentum!,node,R6,b,ny)
   end
  elseif stage==2
   JACC.parallel_for(serial,1,boundary_height!,node,C,b,ny,nx)
  else
   JACC.parallel_for(serial,1,boundary_momentum!,node,C,B,ny,nx,1f-4)
  end
  nothing
 end
end
easywave_study(factory)
