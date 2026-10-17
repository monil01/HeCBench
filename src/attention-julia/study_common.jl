using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function attention_study(factory)
    n=1024;d=64;repeats=10
    key=Float32[Float32(mod(17*i,31)-15)/128f0 for i in 0:n*d-1]
    value=Float32[Float32(mod(19*i,127)-63)/64f0 for i in 0:n*d-1]
    query=Float32[Float32(mod(13*i,17)-8)/64f0 for i in 0:d-1]
    rd=zeros(Float32,n);rs=zeros(Float32,n);ro=zeros(Float32,d);re=0f0
    for i in 0:n-1
        a=0f0;for j in 0:d-1;a+=key[i*d+j+1]*query[j+1];end
        rd[i+1]=a;re+=exp(a)
    end
    for i in 1:n;rs[i]=exp(rd[i])/re;end
    for j in 0:d-1
        for i in 0:n-1;ro[j+1]+=rs[i+1]*value[i*d+j+1];end
    end
    dk=CuArray(key);dv=CuArray(value);dq=CuArray(query);dot=CUDA.zeros(Float32,n);es=CUDA.zeros(Float32,1);score=CUDA.zeros(Float32,n);output=CUDA.zeros(Float32,d)
    launch! = factory(dk,dv,dq,dot,es,score,output)
    OwnedStudy.run_trials!(launch!,()->(fill!(dot,0f0);fill!(es,0f0);fill!(score,0f0);fill!(output,0f0)),()->vcat(Array(dot),Array(es),Array(score),Array(output)),vcat(rd,Float32[re],rs,ro),repeats;atol=1e-3,rtol=1e-6)
end
