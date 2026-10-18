using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function distort_kernel!(src, dst, props)
    prop = @inbounds props[1]
    ix,iy=@index(Global,NTuple)
    w=Int32(ix-1);h=Int32(iy-1)
    if w < prop.width && h < prop.height
        x = radial_x(Float32(w), Float32(h), prop)
        y = radial_y(Float32(w), Float32(h), prop)
        out = h * prop.width + w + Int32(1)
        @inbounds dst[1, out] = interp_channel(src, Int32(1), y, x, prop)
        @inbounds dst[2, out] = interp_channel(src, Int32(2), y, x, prop)
        @inbounds dst[3, out] = interp_channel(src, Int32(3), y, x, prop)
    end
end

function launch(src,dst,props,w,h)
 distort_kernel!(BACKEND,(16,16))(src,dst,props;ndrange=((Int(w)÷16+1)*16,(Int(h)÷16+1)*16))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
