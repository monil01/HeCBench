using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function density_cell_kernel(d,o,m)
 I=@index(Global,NTuple);i=I[1]-1;j=I[2]-1
 if i<96&&j<80;density_cell!(d,o,i,j,m);end
end
@kernel function density_mixed_kernel(d,o)
 c=@index(Global,Linear)-1
 if c<5760;density_mixed!(d,o,c);end
end
@kernel function pressure_cell_kernel(d,o,m)
 I=@index(Global,NTuple);i=I[1]-1;j=I[2]-1
 if i<96&&j<80;pressure_cell!(d,o,i,j,m);end
end
@kernel function pressure_fraction_kernel(d,o)
 ix=@index(Global,Linear)-1
 if ix<17280;pressure_fraction!(d,o,ix);end
end
@kernel function neighbors_cell_kernel(d,o,m)
 I=@index(Global,NTuple);i=I[1]-1;j=I[2]-1
 if 0<i<95&&0<j<79;neighbors_cell!(d,o,i,j,m);end
end
function launch_phase!(phase,d,o,m::Val{M}) where M
 backend=KernelAbstractions.get_backend(d.rho_cell)
 if phase==1
  density_cell_kernel(backend,(32,4))(d,o,m;ndrange=(96,80))
  if M==0;density_mixed_kernel(backend,128)(d,o;ndrange=5760);end
 elseif phase==2
  pressure_cell_kernel(backend,(32,4))(d,o,m;ndrange=(96,80))
  if M==0;pressure_fraction_kernel(backend,128)(d,o;ndrange=17280);end
 else
  neighbors_cell_kernel(backend,(32,4))(d,o,m;ndrange=(96,80))
 end
 return nothing
end
study_main(parse(Int,ARGS[1]))
