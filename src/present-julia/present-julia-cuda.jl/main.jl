using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(num,rounds,plains,keys,ciphers,sbox,t0,t1,t2,t3)
 gid=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1);encrypt_body(gid,num,rounds,plains,keys,ciphers,sbox,t0,t1,t2,t3);return
end
run!(num,rounds,p,k,c,s,t0,t1,t2,t3)=(@cuda threads=256 blocks=cld(num,256) kernel!(num,rounds,p,k,c,s,t0,t1,t2,t3);nothing)
study(run!)
