using CUDA
include(joinpath(@__DIR__,"..","euler_device.jl"))
function initialize_buffer_kernel(d,v,n)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    initialize_buffer_at(i,d,v,n);return nothing
end
function initialize_variables_kernel(n,u,ff)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    initialize_variables_at(i,n,u,ff);return nothing
end
function compute_step_factor_kernel(n,u,a,step)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    compute_step_factor_at(i,n,u,a,step);return nothing
end
function compute_flux_kernel(n,nb,normals,u,ff,flux,de,mx,my,mz)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    compute_flux_at(i,n,nb,normals,u,ff,flux,de,mx,my,mz);return nothing
end
function time_step_kernel(stage,n,old,u,step,flux)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    time_step_at(i,stage,n,old,u,step,flux);return nothing
end
launch_init(n,u,ff)=(@cuda threads=192 blocks=div(n,192) initialize_variables_kernel(n,u,ff);nothing)
launch_zero(n,d)=(@cuda threads=192 blocks=div(n,192) initialize_buffer_kernel(d,0f0,n);nothing)
launch_step(n,u,a,step)=(@cuda threads=192 blocks=div(n,192) compute_step_factor_kernel(n,u,a,step);nothing)
launch_flux(n,nb,normal,u,ff,flux,de,mx,my,mz)=(@cuda threads=192 blocks=div(n,192) compute_flux_kernel(n,nb,normal,u,ff,flux,de,mx,my,mz);nothing)
launch_update(stage,n,old,u,step,flux)=(@cuda threads=192 blocks=div(n,192) time_step_kernel(stage,n,old,u,step,flux);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
