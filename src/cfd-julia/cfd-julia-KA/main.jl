using CUDA, KernelAbstractions
include(joinpath(@__DIR__,"..","euler_device.jl"))
@kernel function initialize_buffer_kernel(d,v,n)
    i=@index(Global,Linear)
    initialize_buffer_at(i,d,v,n);nothing
end
@kernel function initialize_variables_kernel(n,u,ff)
    i=@index(Global,Linear)
    initialize_variables_at(i,n,u,ff);nothing
end
@kernel function compute_step_factor_kernel(n,u,a,step)
    i=@index(Global,Linear)
    compute_step_factor_at(i,n,u,a,step);nothing
end
@kernel function compute_flux_kernel(n,nb,normals,u,ff,flux,de,mx,my,mz)
    i=@index(Global,Linear)
    compute_flux_at(i,n,nb,normals,u,ff,flux,de,mx,my,mz);nothing
end
@kernel function time_step_kernel(stage,n,old,u,step,flux)
    i=@index(Global,Linear)
    time_step_at(i,stage,n,old,u,step,flux);nothing
end
launch_init(n,u,ff)=(initialize_variables_kernel(CUDABackend(),192)(n,u,ff;ndrange=n);nothing)
launch_zero(n,d)=(initialize_buffer_kernel(CUDABackend(),192)(d,0f0,n;ndrange=n);nothing)
launch_step(n,u,a,step)=(compute_step_factor_kernel(CUDABackend(),192)(n,u,a,step;ndrange=n);nothing)
launch_flux(n,nb,normal,u,ff,flux,de,mx,my,mz)=(compute_flux_kernel(CUDABackend(),192)(n,nb,normal,u,ff,flux,de,mx,my,mz;ndrange=n);nothing)
launch_update(stage,n,old,u,step,flux)=(time_step_kernel(CUDABackend(),192)(stage,n,old,u,step,flux;ndrange=n);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
