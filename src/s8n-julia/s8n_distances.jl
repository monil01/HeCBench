using StaticArrays
const RADIUS=Int32(512)
@inline function init_dist8()
    return MVector{8, Int32}(RADIUS, RADIUS, RADIUS, RADIUS, RADIUS, RADIUS, RADIUS, RADIUS)
end

@inline function init_dist16()
    return MVector{16, Int32}(RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS)
end

@inline function init_dist32()
    return MVector{32, Int32}(RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS)
end

