using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function adjacent_study(factory)
    n=1048576;repeats=10
    host=Int32[mod(i,17) for i in 0:n-1];left=similar(host);expected=similar(host)
    for i in 1:n;left[i]=mod(i-1,1024)==0 ? host[i] : host[i]-host[i-1];end
    for i in 1:n;expected[i]=mod(i,1024)==0 ? left[i] : left[i]-left[i+1];end
    input=CuArray(host);output=CUDA.zeros(Int32,n);launch! = factory(input,output)
    OwnedStudy.run_trials!(launch!,()->fill!(output,0),()->Array(output),expected,repeats;atol=0.,rtol=0.)
end
