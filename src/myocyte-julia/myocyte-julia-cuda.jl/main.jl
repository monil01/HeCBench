
using CUDA,Printf
include("../study_equations.jl")
include("../study_common.jl")
function model_kernel(t,y,f,p,com)
 if threadIdx().x==1
  if blockIdx().x==1
   kernel_ecc_jl(t,y,f,Int32(0),p)
  elseif blockIdx().x==2
   kernel_cam_jl(t,y,f,Int32(46),p,Int32(0),com,Int32(0),Float32(y[36]*1e3))
   kernel_cam_jl(t,y,f,Int32(61),p,Int32(5),com,Int32(1),Float32(y[37]*1e3))
   kernel_cam_jl(t,y,f,Int32(76),p,Int32(10),com,Int32(2),Float32(y[38]*1e3))
  end
 end
 nothing
end
myocyte_study(arrays->(t->(@cuda threads=32 blocks=2 model_kernel(t,arrays[1],arrays[3],arrays[2],arrays[4]))))
