using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function cross_ka!(n::Int32,out,a,b)
    row=@index(Global,Linear)-1
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
    kernel=cross_ka!(get_backend(a),256)
    return ()->kernel(Int32(n),out,a,b;ndrange=n)
end
cross_study(factory)
