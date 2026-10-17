using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function ga_study(launch!;args=ARGS)
    n=4096;cl=8;ql=32;threshold=3;repeat=20;state=UInt32(123)
    chars=UInt8.(['A','C','T','G']);bits=UInt8[]
    for _ in 1:n+cl+ql;state=state*UInt32(1664525)+UInt32(1013904223);push!(bits,chars[Int(state>>30)+1]);end
    target=bits[1:n+cl];query=bits[n+cl+1:end];target[1:cl].=query[1:cl]
    expected=zeros(Int32,n)
    for tid in 0:n-1
        for i in 0:ql-cl
            distance=sum(target[tid+j+1]!=query[i+j+1] for j in 0:cl-1)
            if distance<threshold;expected[tid+1]=1;break;end
        end
    end
    any(==(0),expected)&&any(==(1),expected)||error("nontrivial fixture required")
    dt=CuArray(target);dq=CuArray(query);out=CUDA.zeros(UInt8,n)
    function launch()
        for pos in 0:1024:n-1;launch!(dt,dq,view(out,pos+1:pos+1024),UInt32(1024),Int32(ql),Int32(cl),Int32(threshold),Int32(pos));end
    end
    OwnedStudy.run_trials!(launch,()->CUDA.fill!(out,0x00),()->Int32.(Array(out)),expected,repeat;atol=0)
end
