using CUDA,KernelAbstractions
const KERNEL_RADIUS=8
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function conv_rows_kernel!(dst, src, kernel, image_w::Int32, image_h::Int32)
    idx0 = Int32(@index(Global,Linear))-Int32(1)
    total = image_w * image_h
    if idx0 < total
        x0 = idx0 % image_w
        y0 = idx0 ÷ image_w
        sumv = 0.0f0
        k = Int32(-KERNEL_RADIUS)
        while k <= KERNEL_RADIUS
            d = x0 + k
            if d >= 0 && d < image_w
                sumv += @inbounds(src[y0 * image_w + d + Int32(1)]) *
                        @inbounds(kernel[Int32(KERNEL_RADIUS) - k + Int32(1)])
            end
            k += Int32(1)
        end
        @inbounds dst[idx0 + Int32(1)] = sumv
    end
end

@kernel function conv_cols_kernel!(dst, src, kernel, image_w::Int32, image_h::Int32)
    idx0 = Int32(@index(Global,Linear))-Int32(1)
    total = image_w * image_h
    if idx0 < total
        x0 = idx0 % image_w
        y0 = idx0 ÷ image_w
        sumv = 0.0f0
        k = Int32(-KERNEL_RADIUS)
        while k <= KERNEL_RADIUS
            d = y0 + k
            if d >= 0 && d < image_h
                sumv += @inbounds(src[d * image_w + x0 + Int32(1)]) *
                        @inbounds(kernel[Int32(KERNEL_RADIUS) - k + Int32(1)])
            end
            k += Int32(1)
        end
        @inbounds dst[idx0 + Int32(1)] = sumv
    end
end

function factory(out,buffer,input,kernel,w,h)
 kr=conv_rows_kernel!(get_backend(input),256);kc=conv_cols_kernel!(get_backend(input),256)
 return ()->begin
 kr(buffer,input,kernel,Int32(w),Int32(h);ndrange=w*h)
 kc(out,buffer,kernel,Int32(w),Int32(h);ndrange=w*h)
 end
end
conv_study(factory)
