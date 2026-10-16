using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function bilateral_filter_kernel!(input, output, w::Int32, h::Int32,
                                  a_square::Float32, variance_I::Float32,
                                  variance_spatial::Float32, radius::Int32)
    x,y=@index(Global,NTuple)
    idx0=Int32(x)-Int32(1);idy0=Int32(y)-Int32(1)
    if idx0<w && idy0<h

    id = idy0 * w + idx0
    center = input[id + Int32(1)]
    res = Float32(0)
    normalization = Float32(0)

    oi = -radius
    while oi <= radius
        oj = -radius
        while oj <= radius
            idk = idx0 + oi
            idl = idy0 + oj
            if idk < 0
                idk = -idk
            end
            if idl < 0
                idl = -idl
            end
            if idk > w - Int32(1)
                idk = w - Int32(1) - oi
            end
            if idl > h - Int32(1)
                idl = h - Int32(1) - oj
            end

            id_w = idl * w + idk
            I_w = input[id_w + Int32(1)]
            range = -((center - I_w) * (center - I_w)) / (Float32(2) * variance_I)
            spatial = -(Float32((idk - idx0) * (idk - idx0) + (idl - idy0) * (idl - idy0))) /
                      (Float32(2) * variance_spatial)
            weight = a_square * exp(spatial + range)
            normalization += weight
            res += I_w * weight
            oj += Int32(1)
        end
        oi += Int32(1)
    end
    output[id + Int32(1)] = res / normalization
    end
end

function prepare(din,dout,W,H,a,vi,vs)
    kernel=bilateral_filter_kernel!(KernelAbstractions.get_backend(din),(16,16))
    radius->kernel(din,dout,Int32(W),Int32(H),a,vi,vs,radius;ndrange=(W,H))
end
exit(study_main(prepare))
