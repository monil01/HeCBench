using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const POLE = Float32(sqrt(Float32(3.0)) - Float32(2.0))

function pow_two_divider(n::Int32)
    n == 0 && return Int32(0)
    divider = Int32(1)
    while (n & divider) == 0
        divider <<= 1
    end
    return divider
end

function initial_causal_device(image, base::Int32, data_length::Int32, stride::Int32)
    horizon = min(Int32(12), data_length)
    zn = POLE
    sum = @inbounds image[base]
    idx = base
    n = Int32(0)
    while n < horizon
        sum += zn * (@inbounds image[idx])
        zn *= POLE
        idx += stride
        n += Int32(1)
    end
    return sum
end

function convert_coefficients_device!(image, base::Int32, data_length::Int32, stride::Int32)
    lambda = (Float32(1.0) - POLE) * (Float32(1.0) - Float32(1.0) / POLE)
    idx = base
    previous = lambda * initial_causal_device(image, base, data_length, stride)
    @inbounds image[idx] = previous

    n = Int32(1)
    while n < data_length
        idx += stride
        previous = lambda * (@inbounds image[idx]) + POLE * previous
        @inbounds image[idx] = previous
        n += Int32(1)
    end

    previous = (POLE / (POLE - Float32(1.0))) * (@inbounds image[idx])
    @inbounds image[idx] = previous

    n = data_length - Int32(2)
    while n >= 0
        idx -= stride
        previous = POLE * (previous - (@inbounds image[idx]))
        @inbounds image[idx] = previous
        n -= Int32(1)
    end
    return
end

function portable_x!(i,image,width::Int32,height::Int32)
    idx0=Int32(i)-Int32(1)
    if idx0<height
        convert_coefficients_device!(image,idx0*width+Int32(1),width,Int32(1))
    end
    nothing
end
function portable_y!(i,image,width::Int32,height::Int32)
    idx0=Int32(i)-Int32(1)
    if idx0<width
        convert_coefficients_device!(image,idx0+Int32(1),height,width)
    end
    nothing
end
function study_prepare(image,w,h)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=64,blocks=4,shmem_size=0,sync=false)
    function launch!()
        JACC.parallel_for(portable_x!,spec,h,image,Int32(w),Int32(h));JACC.parallel_for(portable_y!,spec,w,image,Int32(w),Int32(h))
    end
end
exit(study_main(study_prepare))
