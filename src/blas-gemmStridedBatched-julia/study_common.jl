using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function batched_study(factory)
    upper=64;m=32;k=32;batch=128;repeats=10
    matrices=Float64[Float64(i%31-15)/17.0 for i in 0:upper*upper*batch-1]
    vectors=Float64[Float64((7*i)%17-8)/19.0 for i in 0:upper*batch-1]
    expected=zeros(Float64,upper*batch)
    for b in 0:batch-1,row in 0:m-1
        value=0.0
        for col in 0:k-1
            value+=matrices[b*upper*upper+row+col*upper+1]*vectors[b*upper+col+1]
        end
        expected[b*upper+row+1]=value
    end
    A=CuArray(matrices);B=CuArray(vectors);C=CUDA.zeros(Float64,upper*batch)
    launch! = factory(A,B,C,m,k,upper,batch)
    OwnedStudy.run_trials!(launch!,()->fill!(C,0.),()->Array(C),expected,repeats;atol=1e-6,rtol=0.)
end
