using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function qkv_kernel(i,src,bias,out,sc,H,D,T)
 qkv_lane(Int32((i-1)÷T),Int32((i-1)%T),src,bias,out,sc,H,D)
end
function factory(s,b,o,sc,m,H,D)
 T=Int32(3H*D÷4);spec=JACC.launch_spec(;threads=Int(T),blocks=Int(m),shmem_size=0,sync=false);()->JACC.parallel_for(spec,Int(m)*Int(T),qkv_kernel,s,b,o,sc,H,D,T)
end
main()=qkv_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
