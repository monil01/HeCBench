using CUDA,JACC
function heat_step!(linear,dest,src,n,r,r2)
 idx=linear-1
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
const SPECS=Dict(n=>JACC.launch_spec(;threads=256,blocks=cld(n*n,256),shmem_size=0,sync=false) for n in [63,127,255])
function launch_jacc(dest,src,n,r,r2)
 JACC.parallel_for(SPECS[Int(n)],cld(Int(n)*Int(n),256)*256,heat_step!,dest,src,n,r,r2)
end
include("../study_common.jl")
heat_study(launch_jacc)
