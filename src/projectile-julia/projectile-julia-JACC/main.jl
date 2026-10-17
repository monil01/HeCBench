using CUDA
import JACC
JACC.@init_backend
function projectile_kernel!(i,angle,velocity,ranges,times,heights,n)
    if i<=n
        @inbounds begin
            s=sin(angle[i]*3.1415f0/180f0);c=cos(angle[i]*3.1415f0/180f0)
            t=abs(2f0*velocity[i]*s)/9.81f0
            ranges[i]=abs(velocity[i]*t*c)
            times[i]=t
            heights[i]=(velocity[i]*velocity[i]*s*s)/2f0*9.81f0
        end
    end
    return
end
const PROJECTILE_SPEC=JACC.launch_spec(threads=256,shmem_size=0,sync=false)
function launch_jacc!(a,v,r,t,h,n)
    JACC.parallel_for(PROJECTILE_SPEC,Int(n),projectile_kernel!,a,v,r,t,h,n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
exit(projectile_study(launch_jacc!))
