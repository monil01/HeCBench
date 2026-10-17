using CUDA
function laplace_kernel!(input,output,nx,ny,nz)
    x=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y)-Int32(1)
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
    (input,output)->(@cuda threads=(32,8) blocks=(cld(nx,32),cld(ny,8)) laplace_kernel!(input,output,Int32(nx),Int32(ny),Int32(nz)))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
