using CUDA,JACC
function bscan_jacc!(lane,out,input,workspace,n)
    shared=JACC.shared(workspace)
    @inbounds value=Int32(input[lane]>0)
    @inbounds shared[lane]=value
    JACC.sync_workgroup()
    offset=1
    while offset<n
        prior=Int32(0)
        if lane>offset;@inbounds prior=shared[lane-offset];end
        JACC.sync_workgroup()
        @inbounds shared[lane]+=prior
        JACC.sync_workgroup()
        offset*=2
    end
    @inbounds out[lane]=shared[lane]-value
    return nothing
end
function launch_jacc!(o,a,s,n)
    spec=JACC.launch_spec(threads=Int(n),shmem_size=4Int(n),sync=false)
    JACC.parallel_for(spec,Int(n),bscan_jacc!,o,a,s,n)
end
include(joinpath(@__DIR__,"..","study_common.jl"));bscan_study(launch_jacc!)
