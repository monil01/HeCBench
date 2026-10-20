using CUDA
import JACC
JACC.@init_backend
include("../device_simulation.jl")
function simulate_kernel!(i,arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    simulate_gate!(Int32(i-1),arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    nothing
end
function launch!(arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    JACC.parallel_for(JACC.launch_spec(threads=32,blocks=Int(cld(n,Int32(32))),shmem_size=0,sync=false),Int(n),simulate_kernel!,arena,func,input,delay,output,capacity,sizes,flags,truth,n,off)
    nothing
end
allocate_data(x)=JACC.array(x)
# The unused arena is not a semantic input; every allocated/live segment is explicitly uploaded.
# JACC.array copies the host allocation here, unlike CUDA/KA direct undef allocation; disclosed covariate.
allocate_arena(T,n)=JACC.array(Vector{T}(undef,n))
include("../study_host.jl")
runstudy()
