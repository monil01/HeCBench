using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function lapl!(g,out,inp,delta,norm,Lx,Ly)
    i=g-1;x=i%Lx;y=i÷Lx
    @inbounds out[g]=norm*inp[g]+delta*(inp[y*Lx+((x+1)%Lx)+1]+inp[y*Lx+((Lx+x-1)%Lx)+1]+inp[((y+1)%Ly)*Lx+x+1]+inp[((Ly+y-1)%Ly)*Lx+x+1])
    return nothing
end
function prepare(a,b,Lx,Ly,delta,norm)
    backend=JACC.get_backend(:cuda)
    spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(Lx*Ly,256),shmem_size=0,sync=false)
    (out,inp)->JACC.parallel_for(lapl!,spec,Lx*Ly,out,inp,delta,norm,Int32(Lx),Int32(Ly))
end
exit(study_main(prepare))
