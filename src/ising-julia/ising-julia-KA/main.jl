using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel cpu=false function init_spins_kernel!(lattice, randvals, nx::Int64, ny::Int64)
    tid0 = Int64(@index(Global,Linear))-1
    total = nx * ny
    if tid0 < total
        @inbounds lattice[tid0 + 1] = randvals[tid0 + 1] < 0.5f0 ? Int8(-1) : Int8(1)
    end
    return
end

@kernel cpu=false function update_lattice_kernel!(lattice, op_lattice, randvals, inv_temp::Float32,
                                nx::Int64, ny::Int64, is_black::Bool)
    tid0 = Int64(@index(Global,Linear))-1
    i = tid0 ÷ ny
    j = tid0 % ny
    if i >= nx || j >= ny
        return
    end
    ipp = i + 1 < nx ? i + 1 : Int64(0)
    inn = i - 1 >= 0 ? i - 1 : nx - 1
    jpp = j + 1 < ny ? j + 1 : Int64(0)
    jnn = j - 1 >= 0 ? j - 1 : ny - 1
    joff = is_black ? (isodd(i) ? jpp : jnn) : (isodd(i) ? jnn : jpp)
    @inbounds nn_sum = op_lattice[inn * ny + j + 1] + op_lattice[i * ny + j + 1] +
                       op_lattice[ipp * ny + j + 1] + op_lattice[i * ny + joff + 1]
    @inbounds lij = lattice[i * ny + j + 1]
    acceptance_ratio = exp(-2.0f0 * inv_temp * Float32(nn_sum) * Float32(lij))
    if randvals[i * ny + j + 1] < acceptance_ratio
        @inbounds lattice[i * ny + j + 1] = -lij
    end
    return
end


function launch_ka(kind,b,w,u,beta,nx,h)
 backend=KernelAbstractions.get_backend(b)
 if kind<2
  init_spins_kernel!(backend,128)(kind==0 ? b : w,u,nx,h;ndrange=nx*h)
 else
  target,other=kind==2 ? (b,w) : (w,b)
  update_lattice_kernel!(backend,128)(target,other,u,beta,nx,h,kind==2;ndrange=nx*h)
 end
end
ising_study(launch_ka)
