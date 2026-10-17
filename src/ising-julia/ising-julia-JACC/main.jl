using CUDA,JACC
include("../study_common.jl")
function init_spins_kernel!(linear,lattice, randvals, nx::Int64, ny::Int64)
    tid0 = Int64(linear)-1
    total = nx * ny
    if tid0 < total
        @inbounds lattice[tid0 + 1] = randvals[tid0 + 1] < 0.5f0 ? Int8(-1) : Int8(1)
    end
    return
end

function update_lattice_kernel!(linear,lattice, op_lattice, randvals, inv_temp::Float32,
                                nx::Int64, ny::Int64, is_black::Bool)
    tid0 = Int64(linear)-1
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


const SPECS=Dict(n=>JACC.launch_spec(;threads=128,blocks=cld(n,128),shmem_size=0,sync=false) for n in (306,1056,4))
function launch_jacc(kind,b,w,u,beta,nx,h)
 n=nx*h;spec=SPECS[n]
 if kind<2
  JACC.parallel_for(spec,n,init_spins_kernel!,kind==0 ? b : w,u,nx,h)
 else
  target,other=kind==2 ? (b,w) : (w,b)
  JACC.parallel_for(spec,n,update_lattice_kernel!,target,other,u,beta,nx,h,kind==2)
 end
end
ising_study(launch_jacc)
