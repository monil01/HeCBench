using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function apd_study(factory)
    instances=224;attributes=4096
    repeats=length(ARGS)>0 ? parse(Int,ARGS[1]) : 10
    host=UInt8[mod((17*k) ⊻ (13*r) ⊻ (k>>3),3) for k in 0:attributes-1,r in 0:instances-1]
    host=vec(host);expected=zeros(Int32,instances*instances)
    for r in 0:instances-1,s in 0:instances-1
        count=Int32(0)
        for k in 0:attributes-1
            count+=host[r*attributes+k+1]!=host[s*attributes+k+1]
        end
        expected[r*instances+s+1]=count
    end
    input=CuArray(host);output=CUDA.zeros(Int32,length(expected))
    launch! = factory(output,input,instances,attributes)
    OwnedStudy.run_trials!(launch!,()->fill!(output,Int32(0)),()->Array(output),expected,repeats)
end
