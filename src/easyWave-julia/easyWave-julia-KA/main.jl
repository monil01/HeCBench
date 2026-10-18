using CUDA,KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function mass!(node,R6,b,ny,zero,arrival,t)
    xy=@index(Global,NTuple)
    j=Int32(xy[1]-1)+b[2];i=Int32(xy[2]-1)+b[1]
    ew_stage1!(node,R6,b...,ny,zero,arrival,t,j,i)
end
@kernel function momentum!(node,R6,b,ny)
    xy=@index(Global,NTuple)
    j=Int32(xy[1]-1)+b[2];i=Int32(xy[2]-1)+b[1]
    ew_stage3!(node,R6,b...,ny,j,i)
end
@kernel function boundary_height!(node,C,b,ny,nx)
    ew_stage2!(node,C...,b...,ny,nx)
end
@kernel function boundary_momentum!(node,C,B,ny,nx,clip)
    ew_stage4!(node,C...,B...,ny,nx,clip)
end
function factory(node,R6,C,B,ny,nx)
 backend=CUDABackend();k1=mass!(backend,(16,16));k2=boundary_height!(backend,1);k3=momentum!(backend,(16,16));k4=boundary_momentum!(backend,1)
 function launch(stage,b,t)
  range=(Int(b[4]-b[2]+1),Int(b[3]-b[1]+1))
  stage==1 && k1(node,R6,b,ny,1f-6,1f-4,t;ndrange=range)
  stage==2 && k2(node,C,b,ny,nx;ndrange=1)
  stage==3 && k3(node,R6,b,ny;ndrange=range)
  stage==4 && k4(node,C,B,ny,nx,1f-4;ndrange=1)
  nothing
 end
end
easywave_study(factory)
