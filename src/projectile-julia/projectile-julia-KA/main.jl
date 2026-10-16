using CUDA, KernelAbstractions
@kernel function projectile_kernel!(angle,velocity,ranges,times,heights,n)
    i=@index(Global,Linear)
    if i<=n
        @inbounds begin
            s=sin(angle[i]*3.1415f0/180f0);c=cos(angle[i]*3.1415f0/180f0)
            t=abs(2f0*velocity[i]*s)/9.81f0
            ranges[i]=abs(velocity[i]*t*c)
            times[i]=t
            heights[i]=(velocity[i]*velocity[i]*s*s)/2f0*9.81f0
        end
    end
end
function launch_ka!(a,v,r,t,h,n)
    projectile_kernel!(get_backend(a),256)(a,v,r,t,h,n;ndrange=Int(n))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
exit(projectile_study(launch_ka!))
