using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
@inline function get_bin(v::Float32, minvalue::Float32, maxvalue::Float32, nbins::Int32)
    bin = Int32(trunc((v - minvalue) * Float32(nbins) / (maxvalue - minvalue)))
    if bin == nbins
        bin -= Int32(1)
    end
    return bin
end

function bincount_study(factory)
    shared=length(ARGS)>0&&ARGS[1]=="shared";n=262144;bins=768;repeats=10
    host=Float32[Float32(mod(i,1025))/1024f0 for i in 0:n-1];expected=zeros(Int32,bins)
    for value in host
        bin=Int(trunc(value*Float32(bins)));bin==bins&&(bin-=1);expected[bin+1]+=Int32(repeats)
    end
    input=CuArray(host);output=CUDA.zeros(Int32,bins);launch! = factory(output,input,shared)
    OwnedStudy.run_trials!(launch!,()->fill!(output,0),()->Array(output),expected,repeats;atol=0.,rtol=0.)
end
