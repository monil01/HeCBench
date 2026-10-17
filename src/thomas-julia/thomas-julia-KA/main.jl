using CUDA,KernelAbstractions
@kernel function thomas_ka!(l,d,u,r,m,n)
 sys=@index(Global,Linear)
 if sys<=n
  @inbounds begin
   first=sys;last=(m-1)*n+sys
   u[first]/=d[first];r[first]/=d[first]
   k=first+n
   while k<last
    den=d[k]-l[k]*u[k-n];u[k]/=den;r[k]=(r[k]-l[k]*r[k-n])/den;k+=n
   end
   r[last]=(r[last]-l[last]*r[last-n])/(d[last]-l[last]*u[last-n])
   k=last-n
   while k>=first;r[k]-=u[k]*r[k+n];k-=n;end
  end
 end
end
function launch_ka(l,d,u,r,m,n)
 thomas_ka!(get_backend(l),128)(l,d,u,r,m,n;ndrange=(Int(n)÷128+1)*128)
end
include("../study_common.jl")
thomas_study(launch_ka)
