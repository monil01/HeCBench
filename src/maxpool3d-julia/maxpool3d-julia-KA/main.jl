using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function maxpool_ka!(input,output,w,h,count,ow,oh)
    xi,yi,zi=@index(Global,NTuple)
    x=xi-1;y=yi-1;z=zi-1
    if x<ow&&y<oh&&z<count
        value=0f0
        for r in Int32(0):Int32(1),c in Int32(0):Int32(1)
            value=max(value,input[((z*h+2*y+r)*w)+2*x+c+1])
        end
        output[(z*oh+y)*ow+x+1]=value
    end
end
function factory(input,output,w,h,count,ow,oh)
    kernel=maxpool_ka!(get_backend(input),(8,8,4))
    return ()->kernel(input,output,w,h,count,ow,oh;ndrange=(Int(ow),Int(oh),Int(count)))
end
pool_study(factory)
