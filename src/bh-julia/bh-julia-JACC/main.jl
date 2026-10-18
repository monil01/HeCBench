import JACC
JACC.@init_backend
include("../study_common.jl")
include("study_kernels.jl")
function factory(nn,nb,steps,start,child,count,sort,p,vel,acc,maxs,mins,radius,bottom,step,bc)
 function run()
  JACC.parallel_for(JACC.launch_spec(threads=1,blocks=1,shmem_size=0,sync=false),1*1,jacc_bh_init!,step,bc)
  for s in 1:steps
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=48,shmem_size=0,sync=false),256*48,jacc_bh_bbox!,nn,nb,start,child,p,maxs,mins,radius,bottom,step,bc)
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=24,shmem_size=0,sync=false),256*24,jacc_bh_clear1!,nn,nb,child)
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=48,shmem_size=0,sync=false),256*48,jacc_bh_build!,nn,nb,child,p,radius,bottom)
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=24,shmem_size=0,sync=false),256*24,jacc_bh_clear2!,nn,start,p,bottom)
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=24,shmem_size=0,sync=false),256*24,jacc_bh_summarize!,nn,nb,count,child,p,bottom)
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=24,shmem_size=0,sync=false),256*24,jacc_bh_sort!,nn,nb,sort,count,start,child,bottom)
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=48,shmem_size=0,sync=false),256*48,jacc_bh_force!,nn,nb,.0125f0,4f0,.0025f0,sort,child,p,vel,acc,radius,step)
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=48,shmem_size=0,sync=false),256*48,jacc_bh_integrate!,nb,.025f0,.0125f0,p,vel,acc)
  end
 end
 return run
end
bh_study(factory,(T,n)->JACC.zeros(T,n))
