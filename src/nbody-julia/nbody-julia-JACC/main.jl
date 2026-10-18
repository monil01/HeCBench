using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function accelerate!(ii,p,n::Int32,soft::Float32,g::Float32)
    i=Int32(ii)-Int32(1)
    accelerate_body!(p,n,soft,g,i)
    return nothing
end
function update!(ii,p,e,n::Int32,dt::Float32)
    i=Int32(ii)-Int32(1)
    update_body!(p,e,n,dt,i)
    return nothing
end
function energy_serial_jacc!(ii,e,n::Int32)
    accumulate_body!(e,n)
    return nothing
end
function launch(phase,p,e,n,dt,soft,g)
    if phase==3
        spec=JACC.launch_spec(;threads=1,blocks=1,shmem_size=0,sync=false)
        JACC.parallel_for(spec,1,energy_serial_jacc!,e,n)
    else
        physical=cld(Int(n),256)*256
        spec=JACC.launch_spec(;threads=256,blocks=physical÷256,shmem_size=0,sync=false)
        if phase==1
            JACC.parallel_for(spec,physical,accelerate!,p,n,soft,g)
        else
            JACC.parallel_for(spec,physical,update!,p,e,n,dt)
        end
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
