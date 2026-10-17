using CUDA
function divergence_kernel!(gv,D,output,temp,rmet,np)
    l=Int32(threadIdx().x)-Int32(1);j=Int32(threadIdx().y)-Int32(1)
    if l<np&&j<np
        dx=0.0;dy=0.0
        @inbounds for i in Int32(0):np-Int32(1)
            dx+=D[l*np+i+Int32(1)]*gv[j*np*Int32(2)+i*Int32(2)+Int32(1)]
            dy+=D[l*np+i+Int32(1)]*gv[i*np*Int32(2)+j*Int32(2)+Int32(2)]
        end
        @inbounds begin
            output[j*np+l+Int32(1)]=dx;temp[l*np+j+Int32(1)]=dy
        end
    end
    CUDA.sync_threads()
    if l<np&&j<np
        idx=l*np+j+Int32(1)
        @inbounds output[idx]=(output[idx]+temp[idx])*(rmet[idx]*1.5683814303638645e-7)
    end
    nothing
end
function study_prepare(gv,D,output,temp,rmet,np)
    @cuda threads=(16,16) blocks=1 divergence_kernel!(gv,D,output,temp,rmet,np)
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
