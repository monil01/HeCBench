using CUDA,KernelAbstractions
@kernel function bscan_ka!(out,input,n)
    lane=@index(Local,Linear)
    shared=@localmem(Int32,(1024,))
    @inbounds value=Int32(input[lane]>0)
    @inbounds shared[lane]=value
    @synchronize
    offset=1
    while offset<n
        prior=Int32(0)
        if lane>offset;@inbounds prior=shared[lane-offset];end
        @synchronize
        @inbounds shared[lane]+=prior
        @synchronize
        offset*=2
    end
    @inbounds out[lane]=shared[lane]-value
end
function launch_ka!(o,a,s,n)
    bscan_ka!(get_backend(o),Int(n))(o,a,n;ndrange=Int(n))
end
include(joinpath(@__DIR__,"..","study_common.jl"));bscan_study(launch_ka!)
