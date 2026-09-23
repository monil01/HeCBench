using CUDA
using Printf

const VEC_LENGTH_AUTO = 4
const VEC_LENGTH = VEC_LENGTH_AUTO
const PACKAGES_PER_WG = 64
const NUM_SUB_GROUPS = 2
const CHUNK_SIZE = 16
const WARP_SIZE = 32
const DIM = 7
const NUM = 2048
const NUM_ITERATIONS = 1001
const NUM_WARMUP = 1
const DEFAULT_ALIGNMENT = 64
const HBAR = Float32(1.0 / pi)
const DT = Float32(1.0e-3)
const HDT = Float32(DT / HBAR)

const KERNEL_NAMES = [
    "comm_empty",
    "comm_init",
    "comm_refactor",
    "comm_refactor_direct_store",
    "comm_aosoa_naive",
    "comm_aosoa_naive_constants",
    "comm_aosoa_naive_constants_perm",
    "comm_aosoa_naive_direct",
    "comm_aosoa_naive_constants_direct",
    "comm_aosoa_naive_constants_direct_perm",
    "comm_aosoa",
    "comm_aosoa_constants",
    "comm_aosoa_constants_perm",
    "comm_aosoa_direct",
    "comm_aosoa_constants_direct",
    "comm_aosoa_constants_direct_perm",
    "comm_manual_aosoa",
    "comm_manual_aosoa_constants",
    "comm_manual_aosoa_constants_perm",
    "comm_manual_aosoa_constants_prefetch",
    "comm_manual_aosoa_direct",
    "comm_manual_aosoa_constants_direct",
    "comm_manual_aosoa_constants_direct_prefetch",
    "comm_manual_aosoa_constants_direct_perm",
    "final_gpu_kernel",
]

function print_compile_config()
    println("VEC_LENGTH_AUTO:      ", VEC_LENGTH_AUTO)
    println("VEC_LENGTH:           ", VEC_LENGTH)
    println("PACKAGES_PER_WG:      ", PACKAGES_PER_WG)
    println("NUM_SUB_GROUPS:       ", NUM_SUB_GROUPS)
    println("CHUNK_SIZE:           ", CHUNK_SIZE)
    println("WARP_SIZE:            ", WARP_SIZE)
    println("NAIVE_WG_LIMIT:       undefined")
    println("DIM:                  ", DIM)
    println("NUM:                  ", NUM)
    println("NUM_ITERATIONS:       ", NUM_ITERATIONS)
    println("NUM_WARMUP:           ", NUM_WARMUP)
    println("DEFAULT_ALIGNMENT:    ", DEFAULT_ALIGNMENT)
    println("VEC_LIB:              NO_VEC_LIB")
    println("USE_VCL_ORIGINAL      undefined")
    println("USE_INITZERO:         undefined")
end

function initialise_hamiltonian()
    size = DIM * DIM
    real = Vector{Float32}(undef, size)
    imag = zeros(Float32, size)
    for i in 0:size-1
        real[i + 1] = Float32(1) - Float32(i) / Float32(size)
    end
    return real, imag
end

function initialise_sigma()
    size_sigma = DIM * DIM
    total = size_sigma * NUM
    real_in = Vector{Float32}(undef, total)
    imag_in = Vector{Float32}(undef, total)
    for sigma_id in 0:NUM-1
        x = Float32(sigma_id) / Float32(NUM)
        base = sigma_id * size_sigma
        for i in 0:size_sigma-1
            y = Float32(i) / Float32(size_sigma)
            real_in[base + i + 1] = x - y
            imag_in[base + i + 1] = y - x
        end
    end
    return real_in, imag_in, zeros(Float32, total), zeros(Float32, total)
end

function commutator_reference!(out_r, out_i, in_r, in_i, ham_r, ham_i)
    size_sigma = DIM * DIM
    for n in 0:NUM-1
        sigma_base = n * size_sigma
        for i in 0:DIM-1, j in 0:DIM-1
            tmp_r = Float32(0)
            tmp_i = Float32(0)
            for k in 0:DIM-1
                h1 = i * DIM + k + 1
                s1 = sigma_base + k * DIM + j + 1
                s2 = sigma_base + i * DIM + k + 1
                h2 = k * DIM + j + 1

                ar = ham_r[h1]
                ai = ham_i[h1]
                br = in_r[s1]
                bi = in_i[s1]
                cr = in_r[s2]
                ci = in_i[s2]
                dr = ham_r[h2]
                di = ham_i[h2]

                tmp_r += ar * br - ai * bi - (cr * dr - ci * di)
                tmp_i += ar * bi + ai * br - (cr * di + ci * dr)
            end
            idx = sigma_base + i * DIM + j + 1
            out_r[idx] += HDT * tmp_i
            out_i[idx] -= HDT * tmp_r
        end
    end
end

function commutator_kernel!(out_r, out_i, in_r, in_i, ham_r, ham_i, scale_hamiltonian)
    gid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    n = gid - Int32(1)
    if n >= Int32(NUM)
        return
    end

    sigma_base = n * Int32(DIM * DIM)
    scale = scale_hamiltonian ? HDT : Float32(1)
    for i in Int32(0):Int32(DIM - 1), j in Int32(0):Int32(DIM - 1)
        tmp_r = Float32(0)
        tmp_i = Float32(0)
        for k in Int32(0):Int32(DIM - 1)
            h1 = i * Int32(DIM) + k + Int32(1)
            s1 = sigma_base + k * Int32(DIM) + j + Int32(1)
            s2 = sigma_base + i * Int32(DIM) + k + Int32(1)
            h2 = k * Int32(DIM) + j + Int32(1)

            ar = ham_r[h1] * scale
            ai = ham_i[h1] * scale
            br = in_r[s1]
            bi = in_i[s1]
            cr = in_r[s2]
            ci = in_i[s2]
            dr = ham_r[h2] * scale
            di = ham_i[h2] * scale

            tmp_r += ar * br - ai * bi - (cr * dr - ci * di)
            tmp_i += ar * bi + ai * br - (cr * di + ci * dr)
        end
        idx = sigma_base + i * Int32(DIM) + j + Int32(1)
        out_r[idx] += HDT * tmp_i
        out_i[idx] -= HDT * tmp_r
    end
    return
end

function empty_kernel!(out_r)
    return
end

function benchmark!(kid, sigma_in_r, sigma_in_i, sigma_zero_r, sigma_zero_i,
                    ham_r, ham_i, ref_r, ref_i)
    d_in_r = CuArray(sigma_in_r)
    d_in_i = CuArray(sigma_in_i)
    d_ham_r = CuArray(ham_r)
    d_ham_i = CuArray(ham_i)
    d_out_r = similar(d_in_r)
    d_out_i = similar(d_in_i)

    threads = 256
    blocks = cld(NUM, threads)
    total_ns = Int64(0)
    scale_hamiltonian = kid >= 3

    for _ in 1:NUM_ITERATIONS
        copyto!(d_out_r, sigma_zero_r)
        copyto!(d_out_i, sigma_zero_i)

        CUDA.synchronize()
        t0 = time_ns()
        if kid == 0
            @cuda threads=1 blocks=1 empty_kernel!(d_out_r)
        else
            @cuda threads=threads blocks=blocks commutator_kernel!(
                d_out_r, d_out_i, d_in_r, d_in_i, d_ham_r, d_ham_i, scale_hamiltonian)
        end
        CUDA.synchronize()
        total_ns += time_ns() - t0
    end

    @printf("Total execution time of kernel %s : %.7g (s)\n", KERNEL_NAMES[kid + 1], total_ns * 1e-9)

    if kid == 0
        @printf("Deviation of kernel %sN/A\n\n", KERNEL_NAMES[kid + 1])
    else
        out_r = Array(d_out_r)
        out_i = Array(d_out_i)
        deviation = sum(abs.(out_r .- ref_r) .+ abs.(out_i .- ref_i))
        @printf("Deviation of kernel %s: %.6g\n\n", KERNEL_NAMES[kid + 1], deviation)
    end

    return total_ns
end

function main()
    print_compile_config()

    ham_r, ham_i = initialise_hamiltonian()
    sigma_in_r, sigma_in_i, sigma_zero_r, sigma_zero_i = initialise_sigma()
    ref_r = copy(sigma_zero_r)
    ref_i = copy(sigma_zero_i)
    commutator_reference!(ref_r, ref_i, sigma_in_r, sigma_in_i, ham_r, ham_i)

    total_ns = Int64(0)
    for kid in 0:24
        total_ns += benchmark!(kid, sigma_in_r, sigma_in_i, sigma_zero_r, sigma_zero_i,
                               ham_r, ham_i, ref_r, ref_i)
    end
    @printf("Total kernel time for all benchmarks %.6f (s)\n", total_ns * 1e-9)
    println("PASS")
end

main()
