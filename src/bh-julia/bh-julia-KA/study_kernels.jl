# Framework-dispatched hybrid CUDA backend bodies; not backend-independent device code.
@kernel function ka_bh_init!(step,bc)
 bh_init!(step,bc)
end

@kernel function ka_bh_bbox!(nn,nb,start,child,p,maxs,mins,radius,bottom,step,bc)
 bh_bbox!(nn,nb,start,child,p,maxs,mins,radius,bottom,step,bc)
end

@kernel function ka_bh_clear1!(nn,nb,child)
 bh_clear1!(nn,nb,child)
end

@kernel function ka_bh_build!(nn,nb,child,p,radius,bottom)
 bh_build!(nn,nb,child,p,radius,bottom)
end

@kernel function ka_bh_clear2!(nn,start,p,bottom)
 bh_clear2!(nn,start,p,bottom)
end

@kernel function ka_bh_summarize!(nn,nb,count,child,p,bottom)
 bh_summarize!(nn,nb,count,child,p,bottom)
end

@kernel function ka_bh_sort!(nn,nb,sort,count,start,child,bottom)
 bh_sort!(nn,nb,sort,count,start,child,bottom)
end

@kernel function ka_bh_force!(nn,nb,halfdt,itolsq,epssq,sort,child,p,vel,acc,radius,step)
 bh_force!(nn,nb,halfdt,itolsq,epssq,sort,child,p,vel,acc,radius,step)
end

@kernel function ka_bh_integrate!(nb,dt,halfdt,p,vel,acc)
 bh_integrate!(nb,dt,halfdt,p,vel,acc)
end
