using CUDA,KernelAbstractions
const MAX_DETECTIONS=Int32(4096)
const N_PARTITIONS=Int32(32)
const OTHRESHOLD=Float32(.3)
include("../study_common.jl")
@kernel function generate_nms_bitmap_kernel!(xs, ys, ws, scores, bitmap, limit::Int32)
    ii,jj=@index(Global,NTuple)
    i0=Int32(ii-1);j0=Int32(jj-1)
    if i0 < limit && j0 < limit
        i = i0 + Int32(1)
        j = j0 + Int32(1)
        if scores[i] < scores[j]
            area = (ws[j] + Float32(1)) * (ws[j] + Float32(1))
            iw = max(Float32(0), min(xs[i] + ws[i], xs[j] + ws[j]) - max(xs[i], xs[j]) + Float32(1))
            ih = max(Float32(0), min(ys[i] + ws[i], ys[j] + ws[j]) - max(ys[i], ys[j]) + Float32(1))
            keep = (((iw * ih) / area) < OTHRESHOLD) && (ws[j] != Float32(0))
            bitmap[Int(i0) * Int(MAX_DETECTIONS) + Int(j0) + 1] = keep ? UInt8(1) : UInt8(0)
        end
    end
end

@kernel function reduce_nms_bitmap_kernel!(bitmap, pointsbitmap, ndetections::Int32)
    row0 = Int32(@index(Group,Linear)-1)
    tx0 = Int32(@index(Local,Linear)-1)
    if row0 < ndetections
        keep = UInt8(1)
        stride = MAX_DETECTIONS ÷ N_PARTITIONS
        for part0 in Int32(0):(N_PARTITIONS - Int32(1))
            col0 = tx0 + part0 * stride
            keep &= bitmap[Int(row0) * Int(MAX_DETECTIONS) + Int(col0) + 1]
        end
        shared = @localmem UInt8 (128,)
        shared[(tx0+1)] = keep
        @synchronize
        step = Int32(64)
        while step > 0
            if tx0 < step
                shared[(tx0+1)] &= shared[(tx0+1) + step]
            end
            @synchronize
            step ÷= Int32(2)
        end
        if tx0 == 0
            pointsbitmap[Int(row0) + 1] = shared[1]
        end
    end
end

function factory(inputs,bitmap,points)
 xs,ys,ws,scores=inputs
 gen=generate_nms_bitmap_kernel!(CUDABackend(),(16,16));red=reduce_nms_bitmap_kernel!(CUDABackend(),128)
 launchmap=()->gen(xs,ys,ws,scores,bitmap,Int32(4096);ndrange=(4096,4096))
 launchreduce=()->red(bitmap,points,Int32(4096);ndrange=4096*128)
 launchmap,launchreduce
end
main()=nms_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
