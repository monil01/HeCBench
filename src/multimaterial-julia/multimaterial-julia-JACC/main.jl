using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function density_cell_kernel(i,j,d,o,m)
 density_cell!(d,o,i-1,j-1,m)
 return nothing
end
function density_mixed_kernel(c,d,o)
 density_mixed!(d,o,c-1)
 return nothing
end
function pressure_cell_kernel(i,j,d,o,m)
 pressure_cell!(d,o,i-1,j-1,m)
 return nothing
end
function pressure_fraction_kernel(ix,d,o)
 pressure_fraction!(d,o,ix-1)
 return nothing
end
function neighbors_cell_kernel(i,j,d,o,m)
 i-=1;j-=1
 if 0<i<95&&0<j<79;neighbors_cell!(d,o,i,j,m);end
 return nothing
end
function launch_phase!(phase,d,o,m::Val{M}) where M
 spec=JACC.launch_spec(threads=(32,4),blocks=(3,20),shmem_size=0,sync=false)
 if phase==1
  JACC.parallel_for(spec,(96,80),density_cell_kernel,d,o,m)
  if M==0;JACC.parallel_for(JACC.launch_spec(threads=128,blocks=45,shmem_size=0,sync=false),5760,density_mixed_kernel,d,o);end
 elseif phase==2
  JACC.parallel_for(spec,(96,80),pressure_cell_kernel,d,o,m)
  if M==0;JACC.parallel_for(JACC.launch_spec(threads=128,blocks=135,shmem_size=0,sync=false),17280,pressure_fraction_kernel,d,o);end
 else
  JACC.parallel_for(spec,(96,80),neighbors_cell_kernel,d,o,m)
 end
 return nothing
end
study_main(parse(Int,ARGS[1]))
