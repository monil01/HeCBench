using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function triad_study(launch!;args=ARGS)
    n=length(args)>0 ? parse(Int,args[1]) : 4194304
    repeats=length(args)>1 ? parse(Int,args[2]) : 100
    n>0 && n%128==0 || error("n must bepositive multiple128")
    a=Float32[Float32(i%1009)/100f0 for i in 0:n-1]
    b=Float32[Float32(i%1013)/100f0 for i in 0:n-1]
    da=CuArray(a);db=CuArray(b);dc=CUDA.zeros(Float32,n)
    launch()=launch!(da,db,dc,1.75f0,n)
    OwnedStudy.run_trials!(launch,()->CUDA.fill!(dc,0f0),()->Array(dc),a.+1.75f0.*b,repeats;atol=1e-3)
end
