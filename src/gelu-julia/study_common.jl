using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
study_gelu(x::Float32)=0.5f0*x*(1f0+tanh(0.79788456f0*(x+0.044715f0*x*x*x)))
function gelu_study(factory)
    mode=length(ARGS)>0 ? ARGS[1] : "base"
    repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10
    mode in ("base","vec2")||error("unknown GELU variant")
    width=1024;height=128;batch=4;total=width*height*batch
    host=Float16[Float32(mod(i,1024))/1024f0 for i in 0:total-1]
    bias=Float16[mod(y,12)-6 for y in 0:width-1]
    expected=copy(host)
    for _ in 1:repeats,i in eachindex(expected)
        y=mod(i-1,width)+1
        summed=Float16(Float32(expected[i])+Float32(bias[y]))
        expected[i]=Float16(study_gelu(Float32(summed)))
    end
    src=CuArray(host);device_bias=CuArray(bias)
    launch! = factory(src,device_bias,mode,width,height,batch)
    # Float16 values exactly represented in Float32 artifacts; widening outsideclock.
    OwnedStudy.run_trials!(launch!,()->copyto!(src,host),()->Float32.(Array(src)),Float32.(expected),repeats;atol=1e-3,rtol=0.)
end
