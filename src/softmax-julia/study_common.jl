using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function softmax_study(factory)
    rows=length(ARGS)>0 ? parse(Int,ARGS[1]) : 8192
    cols=length(ARGS)>1 ? parse(Int,ARGS[2]) : 128
    repeats=length(ARGS)>2 ? parse(Int,ARGS[3]) : 10
    rows>0&&cols>0||error("positive matrix dimensions required")
    host=Float32[mod(17*i,13) for i in 0:rows*cols-1];expected=similar(host)
    for r in 0:rows-1
        offset=r*cols;maximum_value=host[offset+1]
        for j in 1:cols;maximum_value=max(maximum_value,host[offset+j]);end
        total=0f0
        for j in 1:cols;total+=exp(host[offset+j]-maximum_value);end
        for j in 1:cols;expected[offset+j]=exp(host[offset+j]-maximum_value)/total;end
    end
    input=CuArray(host);output=CUDA.zeros(Float32,length(host))
    launch! = factory(output,input,Int32(rows),Int32(cols))
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-3,rtol=0.)
end
