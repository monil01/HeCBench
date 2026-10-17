struct XYZ
    x::Float32
    y::Float32
    z::Float32
end

function bezier_blend(k::Int, mu::Float32, n::Int)
    nn = n
    kn = k
    nkn = n - k
    blend = Float32(1)
    while nn >= 1
        blend *= Float32(nn)
        nn -= 1
        if kn > 1
            blend /= Float32(kn)
            kn -= 1
        end
        if nkn > 1
            blend /= Float32(nkn)
            nkn -= 1
        end
    end
    if k > 0
        blend *= mu ^ Float32(k)
    end
    if n - k > 0
        blend *= (Float32(1) - mu) ^ Float32(n - k)
    end
    return blend
end

function bezier_cpu(inp::Vector{XYZ}, ni::Int, nj::Int, resolution_i::Int, resolution_j::Int)
    outp = Vector{XYZ}(undef, resolution_i * resolution_j)
    for i in 0:(resolution_i - 1)
        mui = Float32(i) / Float32(resolution_i - 1)
        for j in 0:(resolution_j - 1)
            muj = Float32(j) / Float32(resolution_j - 1)
            ox = Float32(0)
            oy = Float32(0)
            oz = Float32(0)
            for ki in 0:ni
                bi = bezier_blend(ki, mui, ni)
                for kj in 0:nj
                    bj = bezier_blend(kj, muj, nj)
                    p = inp[ki * (nj + 1) + kj + 1]
                    w = bi * bj
                    ox += p.x * w
                    oy += p.y * w
                    oz += p.z * w
                end
            end
            outp[i * resolution_j + j + 1] = XYZ(ox, oy, oz)
        end
    end
    return outp
end

