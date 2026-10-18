using CUDA
include("../study_device.jl")
include("../study_common.jl")
function mass!(node,R6,b,ny,zero,arrival,t)
    j=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)+b[2]
    i=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)+b[1]
    ew_stage1!(node,R6,b...,ny,zero,arrival,t,j,i)
    nothing
end
function momentum!(node,R6,b,ny)
    j=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)+b[2]
    i=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)+b[1]
    ew_stage3!(node,R6,b...,ny,j,i)
    nothing
end
function boundary_height!(node,C,b,ny,nx)
    ew_stage2!(node,C...,b...,ny,nx)
    nothing
end
function boundary_momentum!(node,C,B,ny,nx,clip)
    ew_stage4!(node,C...,B...,ny,nx,clip)
    nothing
end
function factory(node,R6,C,B,ny,nx)
 function launch(stage,b,t)
  grid=(cld(Int(b[4]-b[2]+1),16),cld(Int(b[3]-b[1]+1),16))
  stage==1 && CUDA.@cuda(threads=(16,16),blocks=grid,mass!(node,R6,b,ny,1f-6,1f-4,t))
  stage==2 && CUDA.@cuda(threads=1,blocks=1,boundary_height!(node,C,b,ny,nx))
  stage==3 && CUDA.@cuda(threads=(16,16),blocks=grid,momentum!(node,R6,b,ny))
  stage==4 && CUDA.@cuda(threads=1,blocks=1,boundary_momentum!(node,C,B,ny,nx,1f-4))
  nothing
 end
end
easywave_study(factory)
