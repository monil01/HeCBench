using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","quant_math.jl"))
const BACKEND=JACC.get_backend(:cuda)
const TEMPLATE=CUDA.zeros(Float32,1024)
const SPECS=Dict((w,sh)=>JACC.LaunchSpec{typeof(BACKEND)}(threads=w,blocks=17,shmem_size=sh,sync=false) for w in (513,1024) for sh in (0,4096))
@inline function static_scaled_int8_quant_kernel!(i, input, out, scale::Float32, hidden_size::Int32)
    tid0 = ((Int32(i)-Int32(1))%min(hidden_size,Int32(1024))+Int32(1)) - Int32(1)
    token0 = ((Int32(i)-Int32(1))÷min(hidden_size,Int32(1024))+Int32(1)) - Int32(1)
    base = token0 * hidden_size
    i0 = tid0
    @inbounds while i0 < hidden_size
        idx = base + i0 + Int32(1)
        out[idx] = float_to_i8_rn(Float32(input[idx]) / scale)
        i0 += Int32(min(hidden_size,Int32(1024)))
    end
    return
end

@inline function static_scaled_int8_azp_quant_kernel!(i, input, out, scale::Float32, azp::Int32, hidden_size::Int32)
    tid0 = ((Int32(i)-Int32(1))%min(hidden_size,Int32(1024))+Int32(1)) - Int32(1)
    token0 = ((Int32(i)-Int32(1))÷min(hidden_size,Int32(1024))+Int32(1)) - Int32(1)
    base = token0 * hidden_size
    i0 = tid0
    @inbounds while i0 < hidden_size
        idx = base + i0 + Int32(1)
        quant = float_to_i32_rn(Float32(input[idx]) / scale) + azp
        out[idx] = clamp_i8_i32(quant)
        i0 += Int32(min(hidden_size,Int32(1024)))
    end
    return
end

@inline function dynamic_scaled_int8_quant_kernel!(i, input, out, scale, hidden_size::Int32, template)
    tid = ((Int32(i)-Int32(1))%min(hidden_size,Int32(1024))+Int32(1))
    tid0 = tid - Int32(1)
    token0 = ((Int32(i)-Int32(1))÷min(hidden_size,Int32(1024))+Int32(1)) - Int32(1)
    base = token0 * hidden_size
    scratch = JACC.shared(template)

    absmax = Float32(0)
    i0 = tid0
    @inbounds while i0 < hidden_size
        v = abs(Float32(input[base + i0 + Int32(1)]))
        absmax = ifelse(v > absmax, v, absmax)
        i0 += Int32(min(hidden_size,Int32(1024)))
    end
    @inbounds scratch[tid] = absmax
    pad = tid + Int32(min(hidden_size,Int32(1024)))
    @inbounds while pad <= Int32(REDUCE_THREADS)
        scratch[pad] = 0f0
        pad += Int32(min(hidden_size,Int32(1024)))
    end
    JACC.sync_workgroup()

    stride = Int32(REDUCE_THREADS ÷ 2)
    while stride > 0
        if tid <= stride
            other = @inbounds scratch[tid + stride]
            mine = @inbounds scratch[tid]
            @inbounds scratch[tid] = ifelse(other > mine, other, mine)
        end
        JACC.sync_workgroup()
        stride ÷= Int32(2)
    end

    block_absmax = @inbounds scratch[1]
    if tid == Int32(1)
        @inbounds scale[token0 + Int32(1)] = block_absmax / Float32(127)
    end
    JACC.sync_workgroup()

    tmp_scale = Float32(127) / block_absmax
    i0 = tid0
    @inbounds while i0 < hidden_size
        idx = base + i0 + Int32(1)
        out[idx] = float_to_i8_rn(Float32(input[idx]) * tmp_scale)
        i0 += Int32(min(hidden_size,Int32(1024)))
    end
    return
end

@inline function dynamic_scaled_int8_azp_quant_kernel!(i, input, out, scale, azp, hidden_size::Int32, template)
    tid = ((Int32(i)-Int32(1))%min(hidden_size,Int32(1024))+Int32(1))
    tid0 = tid - Int32(1)
    token0 = ((Int32(i)-Int32(1))÷min(hidden_size,Int32(1024))+Int32(1)) - Int32(1)
    base = token0 * hidden_size
    scratch = JACC.shared(template)

    max_val = Float32(floatmin(Float32))
    min_val = Float32(floatmax(Float32))
    i0 = tid0
    @inbounds while i0 < hidden_size
        v = Float32(input[base + i0 + Int32(1)])
        max_val = ifelse(v > max_val, v, max_val)
        min_val = ifelse(v < min_val, v, min_val)
        i0 += Int32(min(hidden_size,Int32(1024)))
    end

    @inbounds scratch[tid] = max_val
    pad = tid + Int32(min(hidden_size,Int32(1024)))
    @inbounds while pad <= Int32(REDUCE_THREADS)
        scratch[pad] = Float32(floatmin(Float32))
        pad += Int32(min(hidden_size,Int32(1024)))
    end
    JACC.sync_workgroup()
    stride = Int32(REDUCE_THREADS ÷ 2)
    while stride > 0
        if tid <= stride
            other = @inbounds scratch[tid + stride]
            mine = @inbounds scratch[tid]
            @inbounds scratch[tid] = ifelse(other > mine, other, mine)
        end
        JACC.sync_workgroup()
        stride ÷= Int32(2)
    end
    block_max = @inbounds scratch[1]
    JACC.sync_workgroup()

    @inbounds scratch[tid] = min_val
    pad = tid + Int32(min(hidden_size,Int32(1024)))
    @inbounds while pad <= Int32(REDUCE_THREADS)
        scratch[pad] = Float32(floatmax(Float32))
        pad += Int32(min(hidden_size,Int32(1024)))
    end
    JACC.sync_workgroup()
    stride = Int32(REDUCE_THREADS ÷ 2)
    while stride > 0
        if tid <= stride
            other = @inbounds scratch[tid + stride]
            mine = @inbounds scratch[tid]
            @inbounds scratch[tid] = ifelse(other < mine, other, mine)
        end
        JACC.sync_workgroup()
        stride ÷= Int32(2)
    end
    block_min = @inbounds scratch[1]

    scale_sh = (block_max - block_min) / Float32(255)
    azp_sh = float_to_i32_rn(Float32(-128) - block_min / scale_sh)
    if tid == Int32(1)
        @inbounds scale[token0 + Int32(1)] = scale_sh
        @inbounds azp[token0 + Int32(1)] = azp_sh
    end
    JACC.sync_workgroup()

    i0 = tid0
    @inbounds while i0 < hidden_size
        idx = base + i0 + Int32(1)
        quant = float_to_i32_rn(Float32(input[idx]) / scale_sh) + azp_sh
        out[idx] = clamp_i8_i32(quant)
        i0 += Int32(min(hidden_size,Int32(1024)))
    end
    return
end


function launch!(p)
 mode,f,di,out,sc,zp=p;h=f[2];nt=f[3];w=min(h,1024);spec=SPECS[(w,mode<=2 ? 0 : 4096)];nd=nt*w
 if mode==1
  JACC.parallel_for(static_scaled_int8_quant_kernel!,spec,nd,di,out,f[4],Int32(h))
 elseif mode==2
  JACC.parallel_for(static_scaled_int8_azp_quant_kernel!,spec,nd,di,out,f[4],f[5],Int32(h))
 elseif mode==3
  JACC.parallel_for(dynamic_scaled_int8_quant_kernel!,spec,nd,di,out,sc,Int32(h),TEMPLATE)
 else
  JACC.parallel_for(dynamic_scaled_int8_azp_quant_kernel!,spec,nd,di,out,sc,zp,Int32(h),TEMPLATE)
 end
end
study_main(launch!)
