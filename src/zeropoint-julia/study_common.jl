using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
@inline function round_nearest_i32(x::Float64)
    return round(Int32, x)
end

@inline function zero_point_one(
    min_in::Float32,
    max_in::Float32,
    qmin::Int32,
    qmax::Int32,
    preserve_sparsity::Bool,
)
    min_val = min_in
    max_val = max_in

    if min_val < 0.0f0 && max_val > 0.0f0 && preserve_sparsity
        symmetric_qmin = -((qmax - qmin) ÷ Int32(2) + Int32(1))
        symmetric_qmax = (qmax - qmin) ÷ Int32(2)
        max_scale = max(abs(Float64(min_val) / Float64(symmetric_qmin)),
                        abs(Float64(max_val) / Float64(symmetric_qmax)))
        min_val = Float32(max_scale * Float64(symmetric_qmin))
        max_val = Float32(max_scale * Float64(symmetric_qmax))
    end

    min_val = min(min_val, 0.0f0)
    max_val = max(max_val, 0.0f0)
    scale = Float32((Float64(max_val) - Float64(min_val)) / Float64(qmax - qmin))

    if scale == 0.0f0 || isinf(1.0f0 / scale)
        scale = 0.1f0
    end

    zero_point_from_min = Float64(qmin) - Float64(min_val) / Float64(scale)
    zero_point_from_max = Float64(qmax) - Float64(max_val) / Float64(scale)
    zero_point_from_min_error = abs(Float64(qmin)) + abs(Float64(min_val) / Float64(scale))
    zero_point_from_max_error = abs(Float64(qmax)) + abs(Float64(max_val) / Float64(scale))
    initial_zero_point = zero_point_from_min_error < zero_point_from_max_error ?
                         zero_point_from_min : zero_point_from_max

    if min_val < 0.0f0 && max_val > 0.0f0 && preserve_sparsity
        initial_zero_point = Float64(qmin + qmax) / 2.0
    end

    nudged = if initial_zero_point < Float64(qmin)
        qmin
    elseif initial_zero_point > Float64(qmax)
        qmax
    else
        round_nearest_i32(initial_zero_point)
    end
    return scale, nudged
end

function zeropoint_study(factory)
    n=262144;repeats=10
    lo=Float32[Float32(mod(17*i,257)-128)/128f0 for i in 0:n-1]
    hi=Float32[Float32(mod(13*i,257)-128)/128f0 for i in 0:n-1]
    expected=Vector{Float64}(undef,2*n)
    for i in 1:n;s,z=zero_point_one(lo[i],hi[i],Int32(-127),Int32(127),true);expected[i]=s;expected[n+i]=z;end
    dl=CuArray(lo);dh=CuArray(hi);scale=CUDA.zeros(Float32,n);zp=CUDA.zeros(Int32,n)
    launch! = factory(dl,dh,scale,zp,Int32(n))
    OwnedStudy.run_trials!(launch!,()->(fill!(scale,0f0);fill!(zp,0)),()->vcat(Float64.(Array(scale)),Float64.(Array(zp))),expected,repeats;atol=1e-3,rtol=0.)
end
