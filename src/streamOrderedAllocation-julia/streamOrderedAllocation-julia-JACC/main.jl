using CUDA,JACC
function vectoradd!(i,a,b,c,n)
 if i<=n;@inbounds c[i]=a[i]+b[i];end
 nothing
end
function launch(a,b,c,n,s)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=cld(Int(n),256),shmem_size=0,sync=false,stream=s)
 JACC.parallel_for(vectoradd!,spec,cld(Int(n),256)*256,a,b,c,n)
 nothing
end
include("../study_common.jl")
allocation_study(launch)
