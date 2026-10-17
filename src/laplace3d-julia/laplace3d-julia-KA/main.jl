using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function laplace_kernel!(input,output,nx,ny,nz)
    ix,iy=@index(Global,NTuple);x=Int32(ix)-Int32(1);y=Int32(iy)-Int32(1)
    if x<nx&&y<ny
        @inbounds for z in Int32(0):nz-Int32(1)
            i=x+y*nx+z*nx*ny+Int32(1)
            if x==0||x==nx-Int32(1)||y==0||y==ny-Int32(1)||z==0||z==nz-Int32(1)
                output[i]=input[i]
            else
                output[i]=(input[i-Int32(1)]+input[i+Int32(1)]+input[i-nx]+input[i+nx]+input[i-nx*ny]+input[i+nx*ny])*(1f0/6f0)
            end
        end
    end
    nothing
end
function study_prepare(nx,ny,nz)
    k=laplace_kernel!(CUDABackend(),(32,8));(input,output)->k(input,output,Int32(nx),Int32(ny),Int32(nz);ndrange=(nx,ny))
end
exit(study_main(study_prepare))
