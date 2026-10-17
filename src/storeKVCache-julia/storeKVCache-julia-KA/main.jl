using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function portable_kv!(kc,vc,loc,k,v,batch::UInt64,row::UInt64)
    idx=UInt64(@index(Global,Linear)-1)
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
    kernel=portable_kv!(CUDABackend(),256);()->kernel(kc,vc,loc,k,v,UInt64(batch),UInt64(row);ndrange=cld(batch,8)*256)
end
exit(study_main(study_prepare))
