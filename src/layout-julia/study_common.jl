using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function layout_study(factory)
    layout=length(ARGS)>0 ? ARGS[1] : "aos"
    repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10
    layout in ("aos","soa")||error("unknown layout")
    trees=4096;size=4096
    host=Int32[r*size+k+mod(r*k,17)-8 for k in 0:size-1,r in 0:trees-1]
    expected=Int32[reinterpret(Int32,UInt32(mod(sum(Int64,view(host,:,r)),Int64(1)<<32))) for r in 1:trees]
    fixture=layout=="aos" ? vec(host) : vec(permutedims(host))
    input=CuArray(fixture);output=CUDA.zeros(Int32,trees)
    launch! = factory(input,output,layout)
    OwnedStudy.run_trials!(launch!,()->fill!(output,Int32(0)),()->Array(output),expected,repeats)
end
