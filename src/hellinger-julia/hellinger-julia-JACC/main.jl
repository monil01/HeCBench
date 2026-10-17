using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function hellinger_kernel!(x,y,a,b,output,rows,terms,cols)
    col=Int32(x)-Int32(1);row=Int32(y)-Int32(1)
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
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(cld(cols,16),cld(rows,16)),shmem_size=0,sync=false)
    ()->JACC.parallel_for(hellinger_kernel!,spec,(cols,rows),a,b,output,Int32(rows),Int32(terms),Int32(cols))
end
exit(study_main(study_prepare))
