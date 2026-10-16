using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function accuracy_study(factory)
    n=4096;d=128;topk=5;repeats=10
    host=Float32[mod(17*i+13*div(i,d),23) for i in 0:n*d-1];labels=Int32[mod(7*i,d) for i in 0:n-1]
    count=Int32(0)
    for row in 0:n-1
        label=labels[row+1];value=host[row*d+label+1];ngt=0
        for col in 0:d-1
            pred=host[row*d+col+1];ngt+=(pred>value||(pred==value&&col<=label))
        end
        count+=ngt<=topk
    end
    input=CuArray(host);dl=CuArray(labels);output=CUDA.zeros(Int32,1)
    kernel! = factory(input,dl,output)
    launch! = ()->begin;fill!(output,0);kernel!();end
    OwnedStudy.run_trials!(launch!,()->fill!(output,0),()->Array(output),Int32[count],repeats;atol=0.,rtol=0.)
end
