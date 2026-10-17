using CUDA,KernelAbstractions
@kernel function heat_step!(dest,src,n,r,r2)
 idx=@index(Global,Linear)-1
 if idx<n*n
  i=idx%n;j=idx÷n;k=idx+1
  @inbounds begin
   right=i<n-1 ? src[k+1] : 0.0
   left=i>0 ? src[k-1] : 0.0
   down=j<n-1 ? src[k+n] : 0.0
   up=j>0 ? src[k-n] : 0.0
   dest[k]=r2*src[k]+r*(right+left+down+up)
  end
 end
end
function launch_ka(dest,src,n,r,r2)
 heat_step!(get_backend(src),256)(dest,src,n,r,r2;ndrange=cld(Int(n)*Int(n),256)*256)
end
include("../study_common.jl")
heat_study(launch_ka)
