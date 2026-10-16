using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function affine_study(factory,reference)
    repeats=length(ARGS)>0 ? parse(Int,ARGS[1]) : 10
    host=UInt16[3*x+5*y for x in 0:511,y in 0:511];host=vec(host)
    expected=Int32.(reference(host))
    input=CuArray(host);output=CUDA.zeros(UInt16,512*512)
    launch! = factory(input,output)
    # Lossless artifact widening UInt16->Int32 only, outside timed computation.
    OwnedStudy.run_trials!(launch!,()->fill!(output,UInt16(0)),()->Int32.(Array(output)),expected,repeats;atol=1.0,rtol=0.)
end
