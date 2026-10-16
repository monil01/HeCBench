using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function copy_study(factory)
    n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 1048576
    repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10
    n>0&&n%1024==0||error("native int4 study requires length divisible by1024 to avoid originalunsignedtailunderflow")
    host=Int32[mod(i,101)-50 for i in 0:n-1]
    input=CuArray(host);output=CUDA.zeros(Int32,n)
    launch! = factory(input,output,Int64(n),Int32(4))
    OwnedStudy.run_trials!(launch!,()->fill!(output,Int32(0)),()->Array(output),host,repeats)
end
