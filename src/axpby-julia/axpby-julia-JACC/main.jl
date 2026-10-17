using CUDA
import JACC
JACC.@init_backend
function axpby_jacc!(index,out,x,y,lengths,offsets,nt,chunk)
    block0=(index-1)÷512
    local0=(index-1)%512
    t=Int32(0);remaining=Int32(block0)
    while t<nt
        chunks=cld(lengths[t+1],chunk)
        remaining<chunks && break
        remaining-=chunks;t+=1
    end
    if t<nt
        len=lengths[t+1];off=offsets[t+1];base=remaining*chunk
        i=Int32(local0)*Int32(4)
        while i<chunk
            for lane in Int32(0):Int32(3)
                k=base+i+lane
                if k<len
                    @inbounds out[off+k+1]=x[off+k+1]+y[off+k+1]
                end
            end
            i+=Int32(2048)
        end
    end
    return
end
const AXPBY_SPEC=JACC.launch_spec(threads=512,shmem_size=0,sync=false)
function launch_jacc!(out,x,y,lengths,offsets,nt,chunk,blocks)
    JACC.parallel_for(AXPBY_SPEC,blocks*512,axpby_jacc!,out,x,y,lengths,offsets,nt,chunk)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
axpby_study(launch_jacc!)
