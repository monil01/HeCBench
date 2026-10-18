include("../study_common.jl")
function factory(nn,nb,steps,start,child,count,sort,p,vel,acc,maxs,mins,radius,bottom,step,bc)
 function run()
  @cuda threads=1 blocks=1 bh_init!(step,bc)
  for s in 1:steps
   @cuda threads=256 blocks=48 bh_bbox!(nn,nb,start,child,p,maxs,mins,radius,bottom,step,bc)
   @cuda threads=256 blocks=24 bh_clear1!(nn,nb,child)
   @cuda threads=256 blocks=48 bh_build!(nn,nb,child,p,radius,bottom)
   @cuda threads=256 blocks=24 bh_clear2!(nn,start,p,bottom)
   @cuda threads=256 blocks=24 bh_summarize!(nn,nb,count,child,p,bottom)
   @cuda threads=256 blocks=24 bh_sort!(nn,nb,sort,count,start,child,bottom)
   @cuda threads=256 blocks=48 bh_force!(nn,nb,.0125f0,4f0,.0025f0,sort,child,p,vel,acc,radius,step)
   @cuda threads=256 blocks=48 bh_integrate!(nb,.025f0,.0125f0,p,vel,acc)
  end
 end
 return run
end
bh_study(factory)
