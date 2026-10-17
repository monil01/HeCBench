using CUDA, KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(num,rounds,plains,keys,ciphers,sbox,t0,t1,t2,t3)
 gid=Int32(@index(Global,Linear))-Int32(1)
 encrypt_body(gid,num,rounds,plains,keys,ciphers,sbox,t0,t1,t2,t3)
end
run!(num,rounds,p,k,c,s,t0,t1,t2,t3)=(kernel!(get_backend(p),256)(num,rounds,p,k,c,s,t0,t1,t2,t3;ndrange=cld(num,256)*256);nothing)
study(run!)
