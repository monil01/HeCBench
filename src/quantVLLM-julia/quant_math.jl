# vLLM-derived quantization helper semantics; see retained LICENSE.
const REDUCE_THREADS=1024
@inline function clamp_i8_i32(x::Int32)
    x < Int32(-128) && return Int8(-128)
    x > Int32(127) && return Int8(127)
    return Int8(x)
end

@inline function float_to_i32_rn(x::Float32)
    x >= Float32(2147483647) && return typemax(Int32)
    x <= Float32(-2147483648) && return typemin(Int32)
    return Int32(round(x))
end

@inline function float_to_i8_rn(x::Float32)
    y = round(x)
    y < Float32(-128) && return Int8(-128)
    y > Float32(127) && return Int8(127)
    return Int8(y)
end

