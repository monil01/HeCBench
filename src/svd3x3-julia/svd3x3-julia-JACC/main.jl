using CUDA
import JACC
JACC.@init_backend
include("../device_common.jl")
allocate_model(a)=JACC.array(a)
function factor_kernel!(i,input,output,n)
    factor_one!(input,output,n,Int32(i)-Int32(1))
    return nothing
end
function launch_svd(w)
    groups=w.n÷256+1
    spec=JACC.launch_spec(threads=256,blocks=groups,shmem_size=0,sync=false)
    JACC.parallel_for(spec,groups*256,factor_kernel!,w.di,w.dout,Int32(w.n))
    nothing
end
include("../study_common.jl")
study_main("jacc")
