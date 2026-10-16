using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function fdtd!(x,y,z,out,inp,coeff,nx,ny,nz,r)
    if x<=nx && y<=ny && z<=nz
        ox=nx+2r;oy=ny+2r;sz=ox*oy
        i=x+r+(y+r-1)*ox+(z+r-1)*sz
        @inbounds begin
            value=coeff[1]*inp[i]
            for k in 1:r
                c=coeff[k+1]
                value+=c*(inp[i+k]+inp[i-k])
                value+=c*(inp[i+k*ox]+inp[i-k*ox])
                value+=c*(inp[i+k*sz]+inp[i-k*sz])
            end
            out[i]=value
        end
    end
    return nothing
end
function prepare(a,b,c,nx,ny,nz,r)
    backend=JACC.get_backend(:cuda)
    spec=JACC.LaunchSpec{typeof(backend)}(;threads=(8,8,4),blocks=(cld(nx,8),cld(ny,8),cld(nz,4)),shmem_size=0,sync=false)
    (out,inp)->JACC.parallel_for(fdtd!,spec,(nx,ny,nz),out,inp,c,Int32(nx),Int32(ny),Int32(nz),Int32(r))
end
exit(study_main(prepare))
