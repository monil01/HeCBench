using CUDA
include("../study_common.jl")
function kernel(x,w,y)
 g=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if g<=512;lstm_lane!(x,w,y,g);end
 return
end
clink_study((x,w,y)->()->(@cuda threads=256 blocks=2 kernel(x,w,y)))
