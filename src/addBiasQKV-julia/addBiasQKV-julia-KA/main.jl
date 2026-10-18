using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function qkv_kernel(src,bias,out,sc,H,D)
 qkv_lane(Int32(@index(Group,Linear))-Int32(1),Int32(@index(Local,Linear))-Int32(1),src,bias,out,sc,H,D)
end
function factory(s,b,o,sc,m,H,D)
 k=qkv_kernel(CUDABackend(),Int(3H*D÷4));()->k(s,b,o,sc,H,D;ndrange=Int(m)*Int(3H*D÷4))
end
main()=qkv_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
