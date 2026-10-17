using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function topk_study(factory)
    batch=128;hidden=4096;k=1024;repeats=10;n=batch*hidden;outsize=batch*k
    host=Float32[mod(5*col+17*row,hidden) for row in 0:batch-1 for col in 0:hidden-1]
    expected=zeros(Float32,2*outsize)
    for row in 0:batch-1
        permutation=sortperm(@view(host[row*hidden+1:(row+1)*hidden]);rev=true)
        for rank in 1:k
            col=permutation[rank];expected[row*k+rank]=host[row*hidden+col];expected[outsize+row*k+rank]=Float32(col-1)
        end
    end
    input=CuArray(host);ids=CUDA.zeros(Int32,outsize);values=CUDA.zeros(Float32,outsize);launch! = factory(input,ids,values)
    OwnedStudy.run_trials!(launch!,()->(fill!(ids,-1);fill!(values,-1f0)),()->vcat(Array(values),Float32.(Array(ids))),expected,repeats;atol=0.,rtol=0.)
end
