include("main.jl")
include("../study_common.jl")
function launch_reference(kind,b,w,u,beta,nx,h)
 if kind<2
  target=kind==0 ? b : w
  @cuda threads=128 blocks=cld(nx*h,128) init_spins_kernel!(target,u,nx,h)
 else
  target,other=kind==2 ? (b,w) : (w,b)
  @cuda threads=128 blocks=cld(nx*h,128) update_lattice_kernel!(target,other,u,beta,nx,h,kind==2)
 end
end
ising_study(launch_reference)
