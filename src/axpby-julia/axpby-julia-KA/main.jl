using CUDA, KernelAbstractions
@kernel function axpby_ka!(out,x,y,lengths,offsets,nt,chunk)
    block0=@index(Group,Linear)-1
    local0=@index(Local,Linear)-1
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
end
function launch_ka!(out,x,y,lengths,offsets,nt,chunk,blocks)
    axpby_ka!(get_backend(out),512)(out,x,y,lengths,offsets,nt,chunk;ndrange=blocks*512)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
axpby_study(launch_ka!)
