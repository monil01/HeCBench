using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,num,rounds,plains,keys,ciphers,sbox,t0,t1,t2,t3)
 encrypt_body(Int32(i)-Int32(1),num,rounds,plains,keys,ciphers,sbox,t0,t1,t2,t3)
 return
end
run!(num,rounds,p,k,c,s,t0,t1,t2,t3)=(JACC.parallel_for(JACC.launch_spec(threads=256,shmem_size=0,sync=false),Int(num),kernel!,num,rounds,p,k,c,s,t0,t1,t2,t3);nothing)
study(run!)
