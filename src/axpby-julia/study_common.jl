using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function axpby_study(launch!;args=ARGS)
    nt=length(args)>0 ? parse(Int,args[1]) : 8
    chunk=length(args)>1 ? parse(Int,args[2]) : 4096
    repeats=length(args)>2 ? parse(Int,args[3]) : 10
    lengths=Int32[1024+((t*7919)%16384) for t in 1:nt]
    offsets=Int32.(vcat(0,cumsum(lengths)[1:end-1]));total=sum(Int.(lengths))
    x=zeros(Float32,total);y=similar(x)
    for t in 0:nt-1,i in 0:Int(lengths[t+1])-1
        k=Int(offsets[t+1])+i+1
        x[k]=Float32((t*17+i*3)%1009);y[k]=Float32((t*29+i*5)%1013)
    end
    dx=CuArray(x);dy=CuArray(y);out=CUDA.zeros(Float32,total);dl=CuArray(lengths);doff=CuArray(offsets)
    blocks=sum(cld(Int(n),chunk) for n in lengths)
    launch()=launch!(out,dx,dy,dl,doff,Int32(nt),Int32(chunk),blocks)
    OwnedStudy.run_trials!(launch,()->CUDA.fill!(out,0f0),()->Array(out),x.+y,repeats;atol=1e-3)
end
