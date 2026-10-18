using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function accelerate!(p,n::Int32,soft::Float32,g::Float32)
    i=(Int32(CUDA.blockIdx().x)-Int32(1))*Int32(256)+Int32(CUDA.threadIdx().x)-Int32(1)
    accelerate_body!(p,n,soft,g,i)
    return
end
function update!(p,e,n::Int32,dt::Float32)
    i=(Int32(CUDA.blockIdx().x)-Int32(1))*Int32(256)+Int32(CUDA.threadIdx().x)-Int32(1)
    update_body!(p,e,n,dt,i)
    return
end
function accumulate!(e,n::Int32)
    accumulate_body!(e,n)
    return
end
function launch(phase,p,e,n,dt,soft,g)
    if phase==1
        CUDA.@cuda threads=256 blocks=cld(Int(n),256) accelerate!(p,n,soft,g)
    elseif phase==2
        CUDA.@cuda threads=256 blocks=cld(Int(n),256) update!(p,e,n,dt)
    else
        CUDA.@cuda threads=1 blocks=1 accumulate!(e,n)
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
