using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function multinomial_study(launch!;args=ARGS)
    nd=1024;nc=64;r=10
    values=Float32[d%17==0 ? 0 : ((cat+d)%4==2 ? 2 : (cat+d)%4==0 ? 0 : 1) for d in 0:nd-1 for cat in 0:nc-1]
    samples=Float32[(2*(d%127)+1)/256 for d in 0:nd-1];expected=zeros(Int32,nd)
    for d in 0:nd-1
        total=sum(values[d*nc+1:(d+1)*nc]);prev=0f0
        if total>0
            for cat in 0:nc-1
                cur=prev+values[d*nc+cat+1]/total
                if prev<=samples[d+1]<cur;expected[d+1]=Int32(cat);break;end
                prev=cur
            end
        end
    end
    ds=CuArray(samples);dv=CuArray(values);out=CUDA.zeros(Int32,nd)
    OwnedStudy.run_trials!(() -> launch!(out,Int32(nd),Int32(nc),ds,dv,Int32(nc),Int32(1)),()->nothing,()->Array(out),expected,r;atol=1)
end
