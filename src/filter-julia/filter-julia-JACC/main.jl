using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_shared!(i,dst,count,src,scratch,n::Int32)
    idx=Int32(i);localidx=(idx-Int32(1))%Int32(256)+Int32(1);counter=JACC.shared(scratch)
    d=Int32(0);pos=Int32(0);keep=false
    if idx<=n
        d=@inbounds src[idx];keep=d>Int32(0)
        if keep
            pos=(JACC.@atomic counter[1] + Int32(1))[1]
        end
    end
    JACC.sync_workgroup()
    if localidx==Int32(1)
        counter[1]=(JACC.@atomic count[1] + counter[1])[1]
    end
    JACC.sync_workgroup()
    if keep
        @inbounds dst[counter[1]+pos+Int32(1)]=d
    end
    JACC.sync_workgroup()
    nothing
end
function portable_global!(i,dst,count,src,n::Int32)
    idx=Int32(i)
    if idx<=n
        d=@inbounds src[idx]
        if d>Int32(0)
            pos=(JACC.@atomic count[1] + Int32(1))[1]
            @inbounds dst[pos+Int32(1)]=d
        end
    end
    nothing
end
function study_prepare(dst,count,src,n)
    scratch=CUDA.zeros(Int32,1);backend=JACC.get_backend(:cuda);ST=JACC.LaunchSpec{typeof(backend)};sharedspec=ST(;threads=256,blocks=cld(n,256),shmem_size=4,sync=false);globalspec=ST(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    function launch!(mode)
        if mode==0
            JACC.parallel_for(portable_shared!,sharedspec,n,dst,count,src,scratch,Int32(n))
        else
            JACC.parallel_for(portable_global!,globalspec,n,dst,count,src,Int32(n))
        end
    end
end
exit(study_main(study_prepare))
