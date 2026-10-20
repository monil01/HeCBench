include(joinpath(@__DIR__,"..","study_common.jl"))
function density_cell_kernel(d,o,m)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 j=(blockIdx().y-1)*blockDim().y+threadIdx().y-1
 if i<96&&j<80;density_cell!(d,o,i,j,m);end
 return nothing
end
function density_mixed_kernel(d,o)
 c=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 if c<5760;density_mixed!(d,o,c);end
 return nothing
end
function pressure_cell_kernel(d,o,m)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 j=(blockIdx().y-1)*blockDim().y+threadIdx().y-1
 if i<96&&j<80;pressure_cell!(d,o,i,j,m);end
 return nothing
end
function pressure_fraction_kernel(d,o)
 ix=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 if ix<17280;pressure_fraction!(d,o,ix);end
 return nothing
end
function neighbors_cell_kernel(d,o,m)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 j=(blockIdx().y-1)*blockDim().y+threadIdx().y-1
 if 0<i<95&&0<j<79;neighbors_cell!(d,o,i,j,m);end
 return nothing
end
function launch_phase!(phase,d,o,m::Val{M}) where M
 if phase==1
  @cuda threads=(32,4) blocks=(3,20) density_cell_kernel(d,o,m)
  if M==0;@cuda threads=128 blocks=45 density_mixed_kernel(d,o);end
 elseif phase==2
  @cuda threads=(32,4) blocks=(3,20) pressure_cell_kernel(d,o,m)
  if M==0;@cuda threads=128 blocks=135 pressure_fraction_kernel(d,o);end
 else
  @cuda threads=(32,4) blocks=(3,20) neighbors_cell_kernel(d,o,m)
 end
 return nothing
end
study_main(parse(Int,ARGS[1]))
