using KernelAbstractions
include("../study_common.jl")
include("study_kernels.jl")
const BACKEND=CUDA.CUDABackend()
function factory(nn,nb,steps,start,child,count,sort,p,vel,acc,maxs,mins,radius,bottom,step,bc)
 function run()
  ka_bh_init!(BACKEND,1)(step,bc;ndrange=1*1)
  for s in 1:steps
   ka_bh_bbox!(BACKEND,256)(nn,nb,start,child,p,maxs,mins,radius,bottom,step,bc;ndrange=256*48)
   ka_bh_clear1!(BACKEND,256)(nn,nb,child;ndrange=256*24)
   ka_bh_build!(BACKEND,256)(nn,nb,child,p,radius,bottom;ndrange=256*48)
   ka_bh_clear2!(BACKEND,256)(nn,start,p,bottom;ndrange=256*24)
   ka_bh_summarize!(BACKEND,256)(nn,nb,count,child,p,bottom;ndrange=256*24)
   ka_bh_sort!(BACKEND,256)(nn,nb,sort,count,start,child,bottom;ndrange=256*24)
   ka_bh_force!(BACKEND,256)(nn,nb,.0125f0,4f0,.0025f0,sort,child,p,vel,acc,radius,step;ndrange=256*48)
   ka_bh_integrate!(BACKEND,256)(nb,.025f0,.0125f0,p,vel,acc;ndrange=256*48)
  end
 end
 return run
end
bh_study(factory,(T,n)->KernelAbstractions.zeros(BACKEND,T,n))
