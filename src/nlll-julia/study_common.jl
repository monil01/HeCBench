using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function nlll_study(factory)
    nframe=4096;kdim=64;repeats=10
    host=Float32[-Float32(mod(17*i,257))/128f0 for i in 0:nframe*kdim-1]
    weights=Float32[Float32(mod(i,7)+1)/8f0 for i in 0:nframe-1]
    target=Int32[mod(i,kdim) for i in 0:nframe-1]
    loss=0f0;weight=0f0
    for i in 0:nframe-1
        t=target[i+1]
        if t!=32;w=weights[t+1];loss-=host[i*kdim+t+1]*w;weight+=w;end
    end
    expected=Float32[loss/weight,weight]
    input=CuArray(host);dw=CuArray(weights);dt=CuArray(target)
    output=CUDA.zeros(Float32,1);total=CUDA.zeros(Float32,1)
    launch! = factory(output,total,input,dt,dw)
    OwnedStudy.run_trials!(launch!,()->(fill!(output,0f0);fill!(total,0f0)),()->vcat(Array(output),Array(total)),expected,repeats;atol=1e-5,rtol=0.)
end
