using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_particle!(i,particle_x,particle_y,random_x,random_y,visits,n_particles::Int32,niterations::Int32,grid_size::Int32,radius::Float32)
    ii0=Int32(i)-Int32(1)
    if ii0<n_particles
    px = @inbounds particle_x[ii0 + Int32(1)]
    py = @inbounds particle_y[ii0 + Int32(1)]
    map_base = ii0 * grid_size * grid_size
    iter = Int32(0)
    while iter < niterations
        randx = @inbounds random_x[iter * n_particles + ii0 + Int32(1)]
        randy = @inbounds random_y[iter * n_particles + ii0 + Int32(1)]
        px += randx / 1000.0f0 - 0.0495f0
        py += randy / 1000.0f0 - 0.0495f0

        dx = px - trunc(px)
        dy = py - trunc(py)
        ix = Int32(floor(px))
        iy = Int32(floor(py))

        if px < Float32(grid_size) && py < Float32(grid_size) && px >= 0.0f0 && py >= 0.0f0
            if dx * dx + dy * dy <= radius * radius
                @inbounds visits[map_base + iy * grid_size + ix + Int32(1)] += UInt64(1)
            end
        end
        iter += Int32(1)
    end

    @inbounds particle_x[ii0 + Int32(1)] = px
    @inbounds particle_y[ii0 + Int32(1)] = py
    end
    nothing
end
function study_prepare(x,y,rx,ry,m,n,steps,g,r)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(portable_particle!,spec,n,x,y,rx,ry,m,Int32(n),Int32(steps),Int32(g),r)
end
exit(study_main(study_prepare))
