using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function sigmoid_jacc2!(ix,iy,mode::Val{P},a,w,h,stride) where P
 x=ix-1;y=iy-1
 if x<w&&y<h
  index=P ? y*stride+x+1 : y*w+x+1
  @inbounds a[index]=1f0/(1f0+exp(-a[index]))
 end
 return
end
@inline function sigmoid_jacc3!(ix,iy,iz,mode::Val{P},a,w,h,d,stride) where P
 x=ix-1;y=iy-1;z=iz-1
 if x<w&&y<h&&z<d
  index=P ? (z*h+y)*stride+x+1 : (z*h+y)*w+x+1
  @inbounds a[index]=1f0/(1f0+exp(-a[index]))
 end
 return
end
function launch!(a,f,pitched)
 w,h,d=Int32.(f[1:3]);stride=Int64(f[5]);backend=JACC.get_backend(:cuda)
 if f[4]
  spec=JACC.LaunchSpec{typeof(backend)}(;threads=(16,8,4),blocks=(cld(w,16),cld(h,8),cld(d,4)),shmem_size=0,sync=false)
  JACC.parallel_for(sigmoid_jacc3!,spec,(Int(w),Int(h),Int(d)),Val(pitched),a,w,h,d,stride)
 else
  spec=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(cld(w,16),cld(h,16)),shmem_size=0,sync=false)
  JACC.parallel_for(sigmoid_jacc2!,spec,(Int(w),Int(h)),Val(pitched),a,w,h,stride)
 end
end
study_main(launch!)
