using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function maxpool_jacc!(xi,yi,zi,input,output,w,h,count,ow,oh)
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
    spec=JACC.launch_spec(;threads=(8,8,4),shmem_size=0,sync=false)
    return ()->JACC.parallel_for(spec,(Int(ow),Int(oh),Int(count)),maxpool_jacc!,input,output,w,h,count,ow,oh)
end
pool_study(factory)
