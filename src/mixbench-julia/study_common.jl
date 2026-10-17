using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function mixbench_study(launch!;args=ARGS)
    n=8*1024*1024;ci=16;repeats=20;threads=256;blocks=n÷8÷128÷threads
    # All workitems have identical initial state. Only first of each8 stridedvalues is overwritten.
    prior=0f0
    for _ in 1:repeats
        a=prior;b=0f0
        for _ in 1:ci;a=a*a+0.1f0;b=b*b+0.1f0;end
        prior=((a*b+b*b)+b*b)+b*b
    end
    expected=zeros(Float32,n)
    for k in 0:3,block in 0:blocks-1,lane in 0:threads-1
        expected[block*threads*8+lane+k*blocks*threads*8+1]=prior
    end
    data=CUDA.zeros(Float32,n)
    OwnedStudy.run_trials!(() -> launch!(data,Int32(ci),blocks),()->CUDA.fill!(data,0f0),()->Array(data),expected,repeats;atol=1e-6)
end
