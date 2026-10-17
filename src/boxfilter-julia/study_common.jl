using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
include(joinpath(@__DIR__,"study_helpers.jl"))
function box_cpu(input,w,h)
    temp=zeros(UInt32,w*h);out=zeros(UInt32,w*h);scale=1f0/21f0
    for y in 0:h-1,x in 0:w-1
        s=(0f0,0f0,0f0,0f0)
        for dx in -10:10
            if 0<=x+dx<w
                s=add4(s,rgba(input[y*w+x+dx+1]))
            end
        end
        temp[y*w+x+1]=pack4(s,scale)
    end
    # Native column kernel never stores row0; preserve that observable reset-zero policy explicitly.
    for x in 0:w-1,y in 1:h-1
        s=(0f0,0f0,0f0,0f0)
        for dy in -10:10
            s=add4(s,rgba(temp[clamp(y+dy,0,h-1)*w+x+1]))
        end
        out[y*w+x+1]=pack4(s,scale)
    end
    out
end
function box_study(factory)
    w=256;h=128;repeats=10
    input=UInt32[UInt32((13*x+7*y)%256)|(UInt32((3*x+11*y)%256)<<8)|(UInt32((17*x+5*y)%256)<<16)|(UInt32((19*x+23*y)%256)<<24) for y in 0:h-1 for x in 0:w-1]
    expected=Int64.(box_cpu(input,w,h));a=CuArray(input);temp=CUDA.zeros(UInt32,w*h);out=CUDA.zeros(UInt32,w*h)
    launch! = factory(a,temp,out,Int32(w),Int32(h),1f0/21f0)
    reset!()=(fill!(temp,UInt32(0));fill!(out,UInt32(0)))
    # Lossless UInt32->Int64 artifact widening outside measured GPU work.
    OwnedStudy.run_trials!(launch!,reset!,()->Int64.(Array(out)),expected,repeats;atol=0.,rtol=0.)
end
