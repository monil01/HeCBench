using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_kv!(i,kc,vc,loc,k,v,batch::UInt64,row::UInt64)
    idx=UInt64(i-1)
    warp=idx÷UInt64(32);lane=idx%UInt64(32)
    if warp<batch
        offset=@inbounds loc[warp+UInt64(1)];srcbase=warp*row;dstbase=offset*row;j=lane
        while j<row
            @inbounds begin
                kc[dstbase+j+UInt64(1)]=k[srcbase+j+UInt64(1)];vc[dstbase+j+UInt64(1)]=v[srcbase+j+UInt64(1)]
            end
            j+=UInt64(32)
        end
    end
    nothing
end
function study_prepare(kc,vc,loc,k,v,batch,row)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(batch,8),shmem_size=0,sync=false)
    ()->JACC.parallel_for(portable_kv!,spec,cld(batch,8)*256,kc,vc,loc,k,v,UInt64(batch),UInt64(row))
end
exit(study_main(study_prepare))
