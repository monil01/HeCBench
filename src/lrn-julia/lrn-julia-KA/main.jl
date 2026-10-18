using CUDA,KernelAbstractions,Printf
const LRN_SIZE=Int32(5);const ALPHA=.000122f0;const BETA=.75f0;const KVAL=1f0
@kernel function lrn_fwd_kernel!(src, dst, total::Int64, cdim::Int32, ddim::Int32, hdim::Int32,
                         wdim::Int32, stride_mb::Int64, size::Int32,
                         alpha::Float32, kval::Float32)
    idx0 = Int64(@index(Global,Linear)-1)
    grid_stride = Int64(@ndrange()[1])
    half_size = (size - Int32(1)) ÷ Int32(2)
    plane = Int64(hdim) * Int64(wdim)
    cube = Int64(ddim) * plane
    sample = Int64(cdim) * cube
    while idx0 < total
        mb = idx0 ÷ sample
        rem = idx0 % sample
        oc = Int32(rem ÷ cube)
        rem %= cube
        od = Int32(rem ÷ plane)
        rem %= plane
        oh = Int32(rem ÷ Int64(wdim))
        ow = Int32(rem % Int64(wdim))

        c_st = max(oc - half_size, Int32(0))
        c_en = min(oc + half_size + Int32(1), cdim)
        sum = 0.0f0
        for c in c_st:(c_en - Int32(1))
            off = mb * stride_mb + Int64(c) * cube + Int64(od) * plane + Int64(oh) * Int64(wdim) + Int64(ow) + 1
            s = @inbounds src[off]
            sum += s * s
        end
        omega = kval + alpha * sum / Float32(size)
        off = idx0 + 1
        s = @inbounds src[off]
        @inbounds dst[off] = s * sqrt(1.0f0 / (sqrt(omega) * omega))
        idx0 += grid_stride
    end
end

@kernel function lrn_bwd_kernel!(src, dst, diff_src, total::Int64, cdim::Int32, ddim::Int32,
                         hdim::Int32, wdim::Int32, stride_mb::Int64,
                         size::Int32, alpha::Float32, beta::Float32, kval::Float32)
    idx0 = Int64(@index(Global,Linear)-1)
    grid_stride = Int64(@ndrange()[1])
    half_size = (size - Int32(1)) ÷ Int32(2)
    plane = Int64(hdim) * Int64(wdim)
    cube = Int64(ddim) * plane
    sample = Int64(cdim) * cube
    while idx0 < total
        mb = idx0 ÷ sample
        rem = idx0 % sample
        oc = Int32(rem ÷ cube)
        rem %= cube
        od = Int32(rem ÷ plane)
        rem %= plane
        oh = Int32(rem ÷ Int64(wdim))
        ow = Int32(rem % Int64(wdim))

        c_st = max(oc - half_size, Int32(0))
        c_en = min(oc + half_size + Int32(1), cdim)
        aval = 0.0f0
        bval = 0.0f0
        for c in c_st:(c_en - Int32(1))
            omega_sum = 0.0f0
            cc_st = max(c - half_size, Int32(0))
            cc_en = min(c + half_size + Int32(1), cdim)
            for cc in cc_st:(cc_en - Int32(1))
                omega_off = mb * stride_mb + Int64(cc) * cube + Int64(od) * plane + Int64(oh) * Int64(wdim) + Int64(ow) + 1
                s = @inbounds src[omega_off]
                omega_sum += s * s
            end
            omega = kval + alpha * omega_sum / Float32(size)
            omega_in_beta = sqrt(1.0f0 / (sqrt(omega) * omega))
            off = mb * stride_mb + Int64(c) * cube + Int64(od) * plane + Int64(oh) * Int64(wdim) + Int64(ow) + 1
            dst_val = @inbounds dst[off]
            tmp = omega_in_beta * dst_val
            if c == oc
                aval = tmp
            end
            src_val = @inbounds src[off]
            bval += src_val * tmp / omega
        end
        off = idx0 + 1
        src_val = @inbounds src[off]
        bval *= 2.0f0 * alpha * beta * src_val / Float32(size)
        @inbounds diff_src[off] = aval - bval
        idx0 += grid_stride
    end
end


include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(src,up,fwd,bwd,dims)
 n,c,d,h,w=dims;total=prod(dims);stride=Int64(c*d*h*w)
 function launch(wg,mode)
  if mode==1
   lrn_fwd_kernel!(CUDABackend(),wg)(src,fwd,Int64(total),Int32(c),Int32(d),Int32(h),Int32(w),stride,LRN_SIZE,ALPHA,KVAL;ndrange=cld(total,wg)*wg)
  else
   lrn_bwd_kernel!(CUDABackend(),wg)(src,up,bwd,Int64(total),Int32(c),Int32(d),Int32(h),Int32(w),stride,LRN_SIZE,ALPHA,BETA,KVAL;ndrange=cld(total,wg)*wg)
  end
 end
end
lrn_study(factory)
