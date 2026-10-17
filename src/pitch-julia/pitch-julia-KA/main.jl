using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function sigmoid_ka2!(mode::Val{P},a,w,h,stride) where P
 ix,iy=@index(Global,NTuple);x=ix-1;y=iy-1
 if x<w&&y<h
  index=P ? y*stride+x+1 : y*w+x+1
  @inbounds a[index]=1f0/(1f0+exp(-a[index]))
 end
end
@kernel function sigmoid_ka3!(mode::Val{P},a,w,h,d,stride) where P
 ix,iy,iz=@index(Global,NTuple);x=ix-1;y=iy-1;z=iz-1
 if x<w&&y<h&&z<d
  index=P ? (z*h+y)*stride+x+1 : (z*h+y)*w+x+1
  @inbounds a[index]=1f0/(1f0+exp(-a[index]))
 end
end
function launch!(a,f,pitched)
 w,h,d=Int32.(f[1:3]);stride=Int64(f[5]);backend=KernelAbstractions.get_backend(a)
 if f[4]
  sigmoid_ka3!(backend,(16,8,4))(Val(pitched),a,w,h,d,stride;ndrange=(Int(w),Int(h),Int(d)))
 else
  sigmoid_ka2!(backend,(16,16))(Val(pitched),a,w,h,stride;ndrange=(Int(w),Int(h)))
 end
end
study_main(launch!)
