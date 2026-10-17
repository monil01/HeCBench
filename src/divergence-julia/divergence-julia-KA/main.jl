using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function divergence_kernel!(gv,D,output,temp,rmet,np)
    x,y=@index(Local,NTuple);l=Int32(x)-Int32(1);j=Int32(y)-Int32(1)
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
    @synchronize
    if l<np&&j<np
        idx=l*np+j+Int32(1)
        @inbounds output[idx]=(output[idx]+temp[idx])*(rmet[idx]*1.5683814303638645e-7)
    end
    nothing
end
const DIV_KERNEL=divergence_kernel!(CUDABackend(),(16,16))
function study_prepare(gv,D,output,temp,rmet,np)
    DIV_KERNEL(gv,D,output,temp,rmet,np;ndrange=(16,16))
end
exit(study_main(study_prepare))
