using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
function distort_kernel!(src, dst, props)
    prop = @inbounds props[1]
    w=Int32(xx-1);h=Int32(yy-1)
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
 blocks=(Int(w)÷16+1,Int(h)÷16+1);spec=JACC.launch_spec(threads=(16,16),blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,(blocks[1]*16,blocks[2]*16),distort_kernel!,src,dst,props)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
