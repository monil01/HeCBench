using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function accelerate!(p,n::Int32,soft::Float32,g::Float32)
    i=Int32(@index(Global,Linear))-Int32(1)
    accelerate_body!(p,n,soft,g,i)
end
@kernel function update!(p,e,n::Int32,dt::Float32)
    i=Int32(@index(Global,Linear))-Int32(1)
    update_body!(p,e,n,dt,i)
end
@kernel function energy_serial_ka!(e,n::Int32)
    accumulate_body!(e,n)
end
function launch(phase,p,e,n,dt,soft,g)
    physical=cld(Int(n),256)*256
    if phase==1
        accelerate!(BACKEND,256)(p,n,soft,g;ndrange=physical)
    elseif phase==2
        update!(BACKEND,256)(p,e,n,dt;ndrange=physical)
    else
        energy_serial_ka!(BACKEND,1)(e,n;ndrange=1)
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
