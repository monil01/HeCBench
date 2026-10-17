using CUDA,KernelAbstractions
@kernel function md_force!(px,py,pz,fx,fy,fz,neighbor_list,n_atom::Int32,max_neighbors::Int32,lj1::Float32,lj2::Float32,cutsq::Float32)
 idx0=Int32(@index(Global,Linear))-Int32(1)
 if idx0<n_atom
    idx = idx0 + Int32(1)
    ipx = @inbounds px[idx]
    ipy = @inbounds py[idx]
    ipz = @inbounds pz[idx]
    ftx = 0.0f0
    fty = 0.0f0
    ftz = 0.0f0

    for j in Int32(0):(max_neighbors - Int32(1))
        jidx0 = @inbounds neighbor_list[Int64(j) * Int64(n_atom) + Int64(idx0) + 1]
        jidx = jidx0 + Int32(1)
        jpx = @inbounds px[jidx]
        jpy = @inbounds py[jidx]
        jpz = @inbounds pz[jidx]

        delx = ipx - jpx
        dely = ipy - jpy
        delz = ipz - jpz
        r2inv = delx * delx + dely * dely + delz * delz
        if r2inv > 0.0f0 && r2inv < cutsq
            r2inv = 1.0f0 / r2inv
            r6inv = r2inv * r2inv * r2inv
            force_c = r2inv * r6inv * (lj1 * r6inv - lj2)
            ftx += delx * force_c
            fty += dely * force_c
            ftz += delz * force_c
        end
    end

    @inbounds begin
        fx[idx] = ftx
        fy[idx] = fty
        fz[idx] = ftz
    end
 end
end
function launch_ka(p,f,neighbors,n)
 md_force!(get_backend(p[1]),256)(p...,f...,neighbors,n,Int32(128),1.5f0,2f0,13.5f0;ndrange=cld(Int(n),256)*256)
end
include("../study_common.jl")
md_study(launch_ka)
