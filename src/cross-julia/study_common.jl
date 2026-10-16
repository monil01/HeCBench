using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function cross_study(factory)
    n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 1048576
    repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10
    n>0||error("positive vector count required")
    a=Float32[(mod(17*i,33)-16)/8f0 for i in 0:3*n-1]
    b=Float32[(mod(13*i,31)-15)/16f0 for i in 0:3*n-1]
    expected=similar(a)
    for r in 0:n-1
        p=3*r+1
        expected[p]=a[p+1]*b[p+2]-a[p+2]*b[p+1]
        expected[p+1]=a[p+2]*b[p]-a[p]*b[p+2]
        expected[p+2]=a[p]*b[p+1]-a[p+1]*b[p]
    end
    da=CuArray(a);db=CuArray(b);output=CUDA.zeros(Float32,length(a))
    launch! = factory(da,db,output,n)
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-6,rtol=0.)
end
