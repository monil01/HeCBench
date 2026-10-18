using CUDA
include("../study_common.jl")
function qkv_kernel(src,bias,out,sc,H,D)
 qkv_lane(Int32(blockIdx().x-1),Int32(threadIdx().x-1),src,bias,out,sc,H,D);nothing
end
main()=qkv_study((s,b,o,sc,m,H,D)->()->(@cuda threads=Int(3H*D÷4) blocks=Int(m) qkv_kernel(s,b,o,sc,H,D)))
if abspath(PROGRAM_FILE)==@__FILE__;main();end
