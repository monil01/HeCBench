include("main.jl")
include("../study_common.jl")
function launch_reference(name,dt,s)
 u,v,F,G,pr,pb,ps,res,mu,mv=s
 if name==:set_BCs
  @cuda threads=16 blocks=2 set_BCs(u,v)
 elseif name==:calculate_F
  @cuda threads=16 blocks=(2,32) calculate_F(dt,u,v,F)
 elseif name==:calculate_G
  @cuda threads=16 blocks=(2,32) calculate_G(dt,u,v,G)
 elseif name==:sum_pressure
  @cuda threads=16 blocks=(1,32) sum_pressure(pr,pb,ps)
 elseif name==:set_horz_pres_BCs
  @cuda threads=16 blocks=1 set_horz_pres_BCs(pr,pb)
 elseif name==:set_vert_pres_BCs
  @cuda threads=16 blocks=1 set_vert_pres_BCs(pr,pb)
 elseif name==:red_kernel
  @cuda threads=16 blocks=(1,32) red_kernel(dt,F,G,pb,pr)
 elseif name==:black_kernel
  @cuda threads=16 blocks=(1,32) black_kernel(dt,F,G,pr,pb)
 elseif name==:calc_residual
  @cuda threads=16 blocks=(1,32) calc_residual(dt,F,G,pr,pb,res)
 elseif name==:calculate_u
  @cuda threads=16 blocks=(1,32) calculate_u(dt,F,pr,pb,u,mu)
 elseif name==:calculate_v
  @cuda threads=16 blocks=(1,32) calculate_v(dt,G,pr,pb,v,mv)
 else;error("Unknown kernel");end
end
cavity_study(launch_reference)
