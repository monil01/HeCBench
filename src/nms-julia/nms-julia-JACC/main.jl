using CUDA,JACC
JACC.@init_backend
const MAX_DETECTIONS=Int32(4096)
const N_PARTITIONS=Int32(32)
const OTHRESHOLD=Float32(.3)
include("../study_common.jl")
function generate_nms_bitmap_kernel!(ii,jj,xs, ys, ws, scores, bitmap, limit::Int32)
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

function reduce_nms_bitmap_kernel!(index, bitmap, pointsbitmap, ndetections::Int32,template)
    row0 = Int32((index-1)÷128)
    tx0 = Int32((index-1)%128)
    if row0 < ndetections
        keep = UInt8(1)
        stride = MAX_DETECTIONS ÷ N_PARTITIONS
        for part0 in Int32(0):(N_PARTITIONS - Int32(1))
            col0 = tx0 + part0 * stride
            keep &= bitmap[Int(row0) * Int(MAX_DETECTIONS) + Int(col0) + 1]
        end
        shared = JACC.shared(template)
        shared[(tx0+1)] = keep
        JACC.sync_workgroup()
        step = Int32(64)
        while step > 0
            if tx0 < step
                shared[(tx0+1)] &= shared[(tx0+1) + step]
            end
            JACC.sync_workgroup()
            step ÷= Int32(2)
        end
        if tx0 == 0
            pointsbitmap[Int(row0) + 1] = shared[1]
        end
    end
end

function factory(inputs,bitmap,points)
 xs,ys,ws,scores=inputs;template=CUDA.zeros(UInt8,128)
 sm=JACC.launch_spec(;threads=(16,16),blocks=(256,256),shmem_size=0,sync=false)
 sr=JACC.launch_spec(;threads=128,blocks=4096,shmem_size=128,sync=false)
 launchmap=()->JACC.parallel_for(sm,(4096,4096),generate_nms_bitmap_kernel!,xs,ys,ws,scores,bitmap,Int32(4096))
 launchreduce=()->JACC.parallel_for(sr,4096*128,reduce_nms_bitmap_kernel!,bitmap,points,Int32(4096),template)
 launchmap,launchreduce
end
main()=nms_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
