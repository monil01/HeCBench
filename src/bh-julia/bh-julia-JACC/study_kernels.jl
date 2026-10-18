# JACC dispatch + explicit CUDA/LLVM backend bodies; no hidden CUDA.@cuda launch.
function jacc_bh_init!(ignored_index,step,bc)
 bh_init!(step,bc)
end

function jacc_bh_bbox!(ignored_index,nn,nb,start,child,p,maxs,mins,radius,bottom,step,bc)
 bh_bbox!(nn,nb,start,child,p,maxs,mins,radius,bottom,step,bc)
end

function jacc_bh_clear1!(ignored_index,nn,nb,child)
 bh_clear1!(nn,nb,child)
end

function jacc_bh_build!(ignored_index,nn,nb,child,p,radius,bottom)
 bh_build!(nn,nb,child,p,radius,bottom)
end

function jacc_bh_clear2!(ignored_index,nn,start,p,bottom)
 bh_clear2!(nn,start,p,bottom)
end

function jacc_bh_summarize!(ignored_index,nn,nb,count,child,p,bottom)
 bh_summarize!(nn,nb,count,child,p,bottom)
end

function jacc_bh_sort!(ignored_index,nn,nb,sort,count,start,child,bottom)
 bh_sort!(nn,nb,sort,count,start,child,bottom)
end

function jacc_bh_force!(ignored_index,nn,nb,halfdt,itolsq,epssq,sort,child,p,vel,acc,radius,step)
 bh_force!(nn,nb,halfdt,itolsq,epssq,sort,child,p,vel,acc,radius,step)
end

function jacc_bh_integrate!(ignored_index,nb,dt,halfdt,p,vel,acc)
 bh_integrate!(nb,dt,halfdt,p,vel,acc)
end
