using CUDA,JACC
const KERNEL_RADIUS=8
include(joinpath(@__DIR__,"..","study_common.jl"))
function conv_rows_kernel!(index,dst, src, kernel, image_w::Int32, image_h::Int32)
    idx0 = Int32(index)-Int32(1)
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
    return
end

function conv_cols_kernel!(index,dst, src, kernel, image_w::Int32, image_h::Int32)
    idx0 = Int32(index)-Int32(1)
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
    return
end

function factory(out,buffer,input,kernel,w,h)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 return ()->begin
 JACC.parallel_for(spec,w*h,conv_rows_kernel!,buffer,input,kernel,Int32(w),Int32(h))
 JACC.parallel_for(spec,w*h,conv_cols_kernel!,out,buffer,kernel,Int32(w),Int32(h))
 end
end
conv_study(factory)
