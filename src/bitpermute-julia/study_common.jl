using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function permutation_study(factory)
    lg=length(ARGS)>0 ? parse(Int,ARGS[1]) : 15
    repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 11
    1<=lg<=20||error("study supports1<=log2size<=20")
    n=1<<lg;host=Int64[i*0x100000001 for i in 0:n-1]
    expected=copy(host)
    if isodd(repeats)
        for i in 0:n-1
            rev=0;x=i
            for _ in 1:lg;rev=(rev<<1)|(x&1);x>>=1;end
            expected[rev+1]=host[i+1]
        end
    end
    device=CuArray(host);launch! = factory(device,lg,n)
    OwnedStudy.run_trials!(launch!,()->copyto!(device,host),()->Array(device),expected,repeats)
end
