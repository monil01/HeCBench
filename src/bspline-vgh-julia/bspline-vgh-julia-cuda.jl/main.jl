using CUDA
include(joinpath(@__DIR__,"..","vgh_math.jl"))
function vgh_kernel!(coefs,vals,grads,hess,w,walker,xs,ys,zs,ix,iy,iz,dx,dy,dz)
    n=Int64((blockIdx().x-1)*blockDim().x+threadIdx().x)
    if n<=2003;vgh_eval!(n,coefs,vals,grads,hess,w,walker,xs,ys,zs,ix,iy,iz,dx,dy,dz);end
    nothing
end
function study_prepare(coefs,vals,grads,hess,w,basis)
    params=ntuple(4) do wi
        wb=(wi-1)*42
        (Int64(basis[wb+37]),Int64(basis[wb+38]),Int64(basis[wb+39]),basis[wb+40],basis[wb+41],basis[wb+42])
    end
    walker->begin
        ix,iy,iz,dx,dy,dz=params[walker+1]
        @cuda threads=256 blocks=8 vgh_kernel!(coefs,vals,grads,hess,w,Int64(walker),Int64(2016*25),Int64(2016*5),Int64(2016),ix,iy,iz,dx,dy,dz)
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
