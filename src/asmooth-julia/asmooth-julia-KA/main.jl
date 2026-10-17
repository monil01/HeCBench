using CUDA
using KernelAbstractions
using Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function linidx0(x::Int32, y::Int32, lx::Int32)
    return y * lx + x + Int32(1)
end

@kernel function smoothing_filter_kernel!(
    lx::Int32,
    ly::Int32,
    threshold::Int32,
    max_rad::Int32,
    img,
    box,
    norm,
)
    ix,iy=@index(Global,NTuple);i=Int32(ix)-Int32(1);j=Int32(iy)-Int32(1)

    if i < lx && j < ly
        gtid = linidx0(i, j, lx)
        sum = 0.0f0
        q = Int32(1)
        s = Int32(1)
        ksum = Int32(0)

        while sum < Float32(threshold) && q < max_rad
            s = q
            sum = 0.0f0
            ksum = Int32(0)

            for ii in -s:s
                for jj in -s:s
                    if (i - s >= Int32(0)) && (i + s < lx) &&
                       (j - s >= Int32(0)) && (j + s < ly)
                        ksum += Int32(1)
                        @inbounds sum += img[linidx0(i + jj, j + ii, lx)]
                    end
                end
            end
            q += Int32(1)
        end

        @inbounds box[gtid] = s
        if ksum != Int32(0)
            inc = 1.0f0 / Float32(ksum)
            for ii in -s:s
                for jj in -s:s
                    if (i - s >= Int32(0)) && (i + s < lx) &&
                       (j - s >= Int32(0)) && (j + s < ly)
                        nidx = linidx0(i + jj, j + ii, lx)
                        Atomix.@atomic norm[nidx] += inc
                    end
                end
            end
        end
    end
    nothing
end

@kernel function normalize_filter_kernel!(lx::Int32, ly::Int32, img, norm)
    ix,iy=@index(Global,NTuple);i=Int32(ix)-Int32(1);j=Int32(iy)-Int32(1)

    if i < lx && j < ly
        gtid = linidx0(i, j, lx)
        @inbounds n = norm[gtid]
        if n != 0.0f0
            @inbounds img[gtid] = img[gtid] / n
        end
    end
    nothing
end

@kernel function out_filter_kernel!(lx::Int32, ly::Int32, img, box, out)
    ix,iy=@index(Global,NTuple);i=Int32(ix)-Int32(1);j=Int32(iy)-Int32(1)

    if i < lx && j < ly
        gtid = linidx0(i, j, lx)
        @inbounds s = box[gtid]
        sum = 0.0f0
        ksum = Int32(0)

        for ii in -s:s
            for jj in -s:s
                if (i - s >= Int32(0)) && (i + s < lx) &&
                   (j - s >= Int32(0)) && (j + s < ly)
                    ksum += Int32(1)
                    @inbounds sum += img[linidx0(i + jj, j + ii, lx)]
                end
            end
        end
        if ksum != Int32(0)
            @inbounds out[gtid] = sum / Float32(ksum)
        end
    end
    nothing
end

function study_prepare(img,box,norm,out,w,t,r)
    backend=CUDABackend();smooth=smoothing_filter_kernel!(backend,(16,16));normalize=normalize_filter_kernel!(backend,(16,16));output=out_filter_kernel!(backend,(16,16))
    function launch!()
        smooth(Int32(w),Int32(w),Int32(t),Int32(r),img,box,norm;ndrange=(w,w));normalize(Int32(w),Int32(w),img,norm;ndrange=(w,w));output(Int32(w),Int32(w),img,box,out;ndrange=(w,w))
    end
end
exit(study_main(study_prepare))
