# Copied byte-identically from original Julia hadamard helper definitions.
@inline function hadamard4x4a(p::Float32, q::Float32, r::Float32, s::Float32)
    t = (p + q + r + s) / Float32(2)
    return p - t, q - t, t - r, t - s
end

@inline function hadamard4x4b(p::Float32, q::Float32, r::Float32, s::Float32)
    t = (p + q + r + s) / Float32(2)
    return t - p, t - q, r - t, s - t
end

