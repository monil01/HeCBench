using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function pool_study(launch_factory)
    w=length(ARGS)>0 ? parse(Int,ARGS[1]) : 256
    h=length(ARGS)>1 ? parse(Int,ARGS[2]) : 256
    count=length(ARGS)>2 ? parse(Int,ARGS[3]) : 16
    repeats=length(ARGS)>3 ? parse(Int,ARGS[4]) : 10
    w>0&&h>0&&count>0&&iseven(w)&&iseven(h)||error("positive even image dimensions required")
    ow=w÷2;oh=h÷2
    host=Float32[(mod(37*i,511)-255)/255f0 for i in 0:w*h*count-1]
    expected=zeros(Float32,ow*oh*count)
    for z in 0:count-1,y in 0:oh-1,x in 0:ow-1
        value=0f0
        for dy in 0:1,dx in 0:1
            value=max(value,host[(z*h+2*y+dy)*w+2*x+dx+1])
        end
        expected[(z*oh+y)*ow+x+1]=value
    end
    input=CuArray(host);output=CUDA.zeros(Float32,length(expected))
    launch! = launch_factory(input,output,Int32(w),Int32(h),Int32(count),Int32(ow),Int32(oh))
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats)
end
