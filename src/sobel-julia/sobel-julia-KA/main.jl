using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function clamp_u8(v::Float32)
    if v > 255.0f0
        return UInt8(255)
    elseif v < 0.0f0
        return UInt8(0)
    else
        return UInt8(unsafe_trunc(Int32, v))
    end
end


@inline study_px(inp,c,off,ch)=Float32(@inbounds inp[Int32(4)*(c+off)+Int32(ch)+Int32(1)])

@kernel function portable_sobel!(inp,out,width::Int32,height::Int32)
    ix,iy=@index(Global,NTuple);x=Int32(ix)-Int32(1);y=Int32(iy)-Int32(1)
    if x >= Int32(1) && x < width - Int32(1) && y >= Int32(1) && y < height - Int32(1)
        c = x + y * width  # 0-based pixel index
        @inbounds begin
            # helper macros not available; inline
            for ch in Int32(0):Int32(3)
                i00 = study_px(inp,c,-Int32(1) - width, ch)
                i01 = study_px(inp,c,-width, ch)
                i02 = study_px(inp,c,Int32(1) - width, ch)
                i10 = study_px(inp,c,-Int32(1), ch)
                i12 = study_px(inp,c,Int32(1), ch)
                i20 = study_px(inp,c,-Int32(1) + width, ch)
                i21 = study_px(inp,c,width, ch)
                i22 = study_px(inp,c,Int32(1) + width, ch)
                Gx = i00 + 2.0f0 * i10 + i20 - i02 - 2.0f0 * i12 - i22
                Gy = i00 - i20 + 2.0f0 * i01 - 2.0f0 * i21 + i02 - i22
                mag = sqrt(Gx * Gx + Gy * Gy) / 2.0f0
                out_idx = Int32(4) * c + ch + Int32(1)
                out[out_idx] = clamp_u8(mag)
            end
        end
    end
end
function study_prepare(din,dout,w,h)
    kernel=portable_sobel!(CUDABackend(),(16,16));()->kernel(din,dout,Int32(w),Int32(h);ndrange=(w,h))
end
exit(study_main(study_prepare))
