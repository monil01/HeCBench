using CUDA,JACC
const X_SIZE=512
const Y_SIZE=512
const PI_F=3.14159265359f0
const WHITE=UInt16(1)
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","study_reference.jl"))
function affine_jacc!(xi,yi,src,dst)
    x=Int32(xi-1);y=Int32(yi-1)

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
        ia00 = 1.0f0; ia01 = 0.0f0; ia10 = 0.0f0; ia11 = 1.0f0
    else
        ia00 =  aff11 / det
        ia01 = -aff01 / det
        ia10 = -aff10 / det
        ia11 =  aff00 / det
    end
    ib0 = 0.0f0
    ib1 = 0.0f0

    xc = Float32(x) - Float32(X_SIZE) / 2.0f0
    yc = Float32(y) - Float32(Y_SIZE) / 2.0f0
    x_new = ib0 + ia00 * xc + ia01 * yc + Float32(X_SIZE) / 2.0f0
    y_new = ib1 + ia10 * xc + ia11 * yc + Float32(Y_SIZE) / 2.0f0

    m = unsafe_trunc(Int32, floor(x_new))
    n = unsafe_trunc(Int32, floor(y_new))

    x_frac = x_new - Float32(m)
    y_frac = y_new - Float32(n)

    output_buffer = WHITE
    if (m >= Int32(0)) && (m + Int32(1) < Int32(X_SIZE)) && (n >= Int32(0)) && (n + Int32(1) < Int32(Y_SIZE))
        @inbounds s0 = Float32(src[n * Int32(X_SIZE) + m + Int32(1)])
        @inbounds s1 = Float32(src[n * Int32(X_SIZE) + m + Int32(2)])
        @inbounds s2 = Float32(src[(n + Int32(1)) * Int32(X_SIZE) + m + Int32(1)])
        @inbounds s3 = Float32(src[(n + Int32(1)) * Int32(X_SIZE) + m + Int32(2)])
        gray_new = (1.0f0 - y_frac) * ((1.0f0 - x_frac) * s0 + x_frac * s1) +
                          y_frac  * ((1.0f0 - x_frac) * s2 + x_frac * s3)
        output_buffer = unsafe_trunc(UInt16, gray_new)
    elseif ((m + Int32(1) == Int32(X_SIZE)) && (n >= Int32(0)) && (n < Int32(Y_SIZE))) ||
           ((n + Int32(1) == Int32(Y_SIZE)) && (m >= Int32(0)) && (m < Int32(X_SIZE)))
        @inbounds output_buffer = src[n * Int32(X_SIZE) + m + Int32(1)]
    end

    if x >= Int32(0) && x < Int32(X_SIZE) && y >= Int32(0) && y < Int32(Y_SIZE)
        @inbounds dst[y * Int32(X_SIZE) + x + Int32(1)] = output_buffer
    end
end

function factory(input,output)
    spec=JACC.launch_spec(;threads=(16,16),sync=false)
    return ()->JACC.parallel_for(spec,(512,512),affine_jacc!,input,output)
end
affine_study(factory,affine_reference)
