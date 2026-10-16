using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function permutation_jacc!(index,buffer,lg::UInt32,n::UInt32)
    i=UInt32(index-1)
    if i<n
        x=i;rev=UInt32(0)
        for _ in UInt32(0):UInt32(31)
            rev=(rev<<UInt32(1))|(x&UInt32(1));x>>=UInt32(1)
        end
        rev>>=UInt32(32)-lg
        if i<rev
            a=buffer[Int(i)+1];b=buffer[Int(rev)+1]
            buffer[Int(i)+1]=b;buffer[Int(rev)+1]=a
        end
    end
end
function factory(buffer,lg,n)
    spec=JACC.launch_spec(;threads=256,sync=false)
    return ()->JACC.parallel_for(spec,n,permutation_jacc!,buffer,UInt32(lg),UInt32(n))
end
permutation_study(factory)
