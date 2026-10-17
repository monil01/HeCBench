using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function stddev_study(factory)
    d=256;n=4096;repeats=10
    host=Float32[Float32(mod(i,1024))/1024f0 for i in 0:d*n-1];expected=zeros(Float32,d)
    for col in 0:d-1
        total=0f0
        for row in 0:n-1;v=host[row*d+col+1];total+=v*v;end
        expected[col+1]=sqrt(total/Float32(n-1))
    end
    input=CuArray(host);output=CUDA.zeros(Float32,d);launch! = factory(output,input)
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-3,rtol=0.)
end
