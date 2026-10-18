using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function distort_kernel!(src, dst, props)
    prop = @inbounds props[1]
    w = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    h = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if w < prop.width && h < prop.height
        x = radial_x(Float32(w), Float32(h), prop)
        y = radial_y(Float32(w), Float32(h), prop)
        out = h * prop.width + w + Int32(1)
        @inbounds dst[1, out] = interp_channel(src, Int32(1), y, x, prop)
        @inbounds dst[2, out] = interp_channel(src, Int32(2), y, x, prop)
        @inbounds dst[3, out] = interp_channel(src, Int32(3), y, x, prop)
    end
    return
end

function launch(src,dst,props,w,h)
 @cuda threads=(16,16) blocks=(Int(w)÷16+1,Int(h)÷16+1) distort_kernel!(src,dst,props)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
