# CPU reference preserved from existing CUDA.jl source for independent host checks.
function affine_reference(src)
    dst = Vector{UInt16}(undef, X_SIZE * Y_SIZE)
    lx_rot = 30.0f0
    ly_rot = 0.0f0
    lx_expan = 0.5f0
    ly_expan = 0.5f0
    aff00 = lx_expan * cos(lx_rot * PI_F / 180.0f0)
    aff01 = ly_expan * sin(ly_rot * PI_F / 180.0f0)
    aff10 = lx_expan * sin(lx_rot * PI_F / 180.0f0)
    aff11 = ly_expan * cos(ly_rot * PI_F / 180.0f0)
    det = aff00 * aff11 - aff01 * aff10
    if det == 0.0f0
        ia00, ia01, ia10, ia11 = 1f0, 0f0, 0f0, 1f0
    else
        ia00 =  aff11 / det
        ia01 = -aff01 / det
        ia10 = -aff10 / det
        ia11 =  aff00 / det
    end
    for y in 0:Y_SIZE-1
        for x in 0:X_SIZE-1
            xc = Float32(x) - Float32(X_SIZE) / 2.0f0
            yc = Float32(y) - Float32(Y_SIZE) / 2.0f0
            x_new = ia00 * xc + ia01 * yc + Float32(X_SIZE) / 2.0f0
            y_new = ia10 * xc + ia11 * yc + Float32(Y_SIZE) / 2.0f0
            m = Int(floor(x_new))
            n = Int(floor(y_new))
            x_frac = x_new - Float32(m)
            y_frac = y_new - Float32(n)
            ob = WHITE
            if (m >= 0) && (m + 1 < X_SIZE) && (n >= 0) && (n + 1 < Y_SIZE)
                s0 = Float32(src[n*X_SIZE + m + 1])
                s1 = Float32(src[n*X_SIZE + m + 2])
                s2 = Float32(src[(n+1)*X_SIZE + m + 1])
                s3 = Float32(src[(n+1)*X_SIZE + m + 2])
                gray_new = (1f0 - y_frac) * ((1f0 - x_frac) * s0 + x_frac * s1) +
                                y_frac  * ((1f0 - x_frac) * s2 + x_frac * s3)
                ob = unsafe_trunc(UInt16, gray_new)
            elseif ((m + 1 == X_SIZE) && (n >= 0) && (n < Y_SIZE)) ||
                   ((n + 1 == Y_SIZE) && (m >= 0) && (m < X_SIZE))
                ob = src[n*X_SIZE + m + 1]
            end
            dst[y*X_SIZE + x + 1] = ob
        end
    end
    return dst
end

