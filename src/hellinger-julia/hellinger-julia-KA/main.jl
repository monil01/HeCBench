using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function hellinger_kernel!(a,b,output,rows,terms,cols)
    x,y=@index(Global,NTuple);col=Int32(x)-Int32(1);row=Int32(y)-Int32(1)
    if row<rows&&col<cols
        s=0f0
        @inbounds for i in Int32(0):terms-Int32(1)
            s+=sqrt(a[row*terms+i+Int32(1)]*b[i*cols+col+Int32(1)])
        end
        value=1f0-s;gate=signbit(value) ? 0f0 : 1f0
        @inbounds output[row*cols+col+Int32(1)]=sqrt(gate*value)
    end
    nothing
end
function study_prepare(a,b,output,rows,terms,cols)
    k=hellinger_kernel!(CUDABackend(),(16,16))
    ()->k(a,b,output,Int32(rows),Int32(terms),Int32(cols);ndrange=(cols,rows))
end
exit(study_main(study_prepare))
