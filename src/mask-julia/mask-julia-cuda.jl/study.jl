include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(a,seq,centers,out,f,::Val{M}) where M
 m,n,barg,b,rad,blocks=f[1:6];ni=Int32(n);mi=Int32(m);bi=Int32(b)
 if M==1
  @cuda threads=256 blocks=blocks sequence_mask_kernel!(ni,mi,bi,a,seq,Int32(-1),out)
 elseif M==2
  @cuda threads=256 blocks=blocks window_mask_kernel!(ni,mi,bi,a,centers,Int32(rad),Int32(-1),out)
 elseif M==3
  @cuda threads=256 blocks=blocks upper_mask_kernel!(ni,mi,bi,a,Int32(-1),out)
 elseif M==4
  @cuda threads=256 blocks=blocks lower_mask_kernel!(ni,mi,bi,a,Int32(-1),out)
 elseif M==5
  @cuda threads=256 blocks=blocks upper_diag_mask_kernel!(ni,mi,bi,a,Int32(-1),out)
 else
  @cuda threads=256 blocks=blocks lower_diag_mask_kernel!(ni,mi,bi,a,Int32(-1),out)
 end
end
study_main(study_launch!)
