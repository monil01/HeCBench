using CUDA
function hellinger_kernel!(a,b,output,rows,terms,cols)
    col=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1);row=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y)-Int32(1)
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
    ()->(@cuda threads=(16,16) blocks=(cld(cols,16),cld(rows,16)) hellinger_kernel!(a,b,output,Int32(rows),Int32(terms),Int32(cols)))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
