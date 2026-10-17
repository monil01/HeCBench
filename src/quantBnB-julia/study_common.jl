using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function load_codebook()
    src = read(joinpath(@__DIR__, "..", "quantBnB-cuda", "code.h"), String)
    m = match(r"\{(.*)\}", src)
    m === nothing && error("could not parse quantization codebook")
    return Float32.(parse.(Float64, split(m.captures[1], ",")))
end

@inline function dquantize(code, x::Float32)
    pivot = 128
    upper_pivot = 256
    lower_pivot = 1
    lower = Float32(-1)
    upper = Float32(1)
    val = @inbounds code[pivot]

    step = 64
    while step > 0
        if x > val
            lower_pivot = pivot
            lower = val
            pivot += step
        else
            upper_pivot = pivot
            upper = val
            pivot -= step
        end
        val = @inbounds code[pivot]
        step >>>= 1
    end

    if upper_pivot == 256
        upper = @inbounds code[upper_pivot]
    end
    if lower_pivot == 1
        lower = @inbounds code[lower_pivot]
    end

    if x > val
        midpoint = (upper + val) * Float32(0.5)
        return UInt8(x > midpoint ? upper_pivot - 1 : pivot - 1)
    else
        midpoint = (lower + val) * Float32(0.5)
        return UInt8(x < midpoint ? lower_pivot - 1 : pivot - 1)
    end
end

function quantbnb_study(factory)
    n=262144;repeats=10;code=load_codebook()
    host=Float32[Float32(mod(17*i,4097)-2048)/1024f0 for i in 0:n-1]
    expected=Int32[dquantize(code,x) for x in host]
    dc=CuArray(code);input=CuArray(host);output=CUDA.zeros(UInt8,n);launch! = factory(dc,input,output)
    OwnedStudy.run_trials!(launch!,()->fill!(output,0x00),()->Int32.(Array(output)),expected,repeats;atol=0.,rtol=0.)
end
