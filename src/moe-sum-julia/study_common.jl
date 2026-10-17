using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function moe_study(factory)
    vec=length(ARGS)>0&&ARGS[1]=="vec4";tokens=512;hidden=1024;topk=3;repeats=10;n=tokens*hidden
    host=Float32[Float32(mod(17*i,257)-128)/128f0 for i in 0:n*topk-1];expected=zeros(Float32,n)
    for token in 0:tokens-1,j in 0:hidden-1,k in 0:topk-1;expected[token*hidden+j+1]+=host[token*hidden*topk+k*hidden+j+1];end
    input=CuArray(host);output=CUDA.zeros(Float32,n);launch! = factory(input,output,vec)
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-4,rtol=0.)
end
