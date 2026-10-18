include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(p)
 mode,f,di,out,sc,zp=p;h=f[2];nt=f[3];threads=min(h,1024)
 if mode==1
  @cuda threads=threads blocks=nt static_scaled_int8_quant_kernel!(di,out,f[4],Int32(h))
 elseif mode==2
  @cuda threads=threads blocks=nt static_scaled_int8_azp_quant_kernel!(di,out,f[4],f[5],Int32(h))
 elseif mode==3
  @cuda threads=threads blocks=nt dynamic_scaled_int8_quant_kernel!(di,out,sc,Int32(h))
 else
  @cuda threads=threads blocks=nt dynamic_scaled_int8_azp_quant_kernel!(di,out,sc,zp,Int32(h))
 end
end
study_main(launch!)
