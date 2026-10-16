using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function cross_jacc!(index,n::Int32,out,a,b)
    row=index-1
    if row<n
        p=3*row+1
        a0=a[p];a1=a[p+1];a2=a[p+2]
        b0=b[p];b1=b[p+1];b2=b[p+2]
        out[p]=a1*b2-a2*b1
        out[p+1]=a2*b0-a0*b2
        out[p+2]=a0*b1-a1*b0
    end
end
function factory(a,b,out,n)
    spec=JACC.launch_spec(;threads=256,sync=false)
    return ()->JACC.parallel_for(spec,n,cross_jacc!,Int32(n),out,a,b)
end
cross_study(factory)
