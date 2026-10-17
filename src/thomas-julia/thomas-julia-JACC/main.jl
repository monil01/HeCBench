using CUDA,JACC
function thomas_jacc!(sys,l,d,u,r,m,n)
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
function launch_jacc(l,d,u,r,m,n)
 blocks=Int(n)÷128+1
 spec=JACC.launch_spec(;threads=128,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*128,thomas_jacc!,l,d,u,r,m,n)
end
include("../study_common.jl")
thomas_study(launch_jacc)
