include("main.jl")
include("../study_common.jl")
function factory(inputs,bitmap,points)
 xs,ys,ws,scores=inputs
 map=()->begin;@cuda threads=(16,16) blocks=(256,256) generate_nms_bitmap_kernel!(xs,ys,ws,scores,bitmap,Int32(4096));end
 reduce=()->begin;@cuda threads=128 blocks=4096 reduce_nms_bitmap_kernel!(bitmap,points,Int32(4096));end
 map,reduce
end
nms_study(factory)
