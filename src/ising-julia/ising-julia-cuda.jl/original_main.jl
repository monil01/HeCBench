using CUDA
using Printf

const TCRIT = 2.26918531421f0
const THREADS = 128

function init_spins_kernel!(lattice, randvals, nx::Int64, ny::Int64)
    tid0 = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    total = nx * ny
    if tid0 < total
        @inbounds lattice[tid0 + 1] = randvals[tid0 + 1] < 0.5f0 ? Int8(-1) : Int8(1)
    end
    return
end

function update_lattice_kernel!(lattice, op_lattice, randvals, inv_temp::Float32,
                                nx::Int64, ny::Int64, is_black::Bool)
    tid0 = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
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

function init_spins_ref!(lattice, randvals, nx, ny)
    for tid in 0:nx*ny-1
        lattice[tid + 1] = randvals[tid + 1] < 0.5f0 ? Int8(-1) : Int8(1)
    end
end

function update_lattice_ref!(lattice, op_lattice, randvals, inv_temp, nx, ny, is_black)
    for i in 0:nx-1, j in 0:ny-1
        ipp = i + 1 < nx ? i + 1 : 0
        inn = i - 1 >= 0 ? i - 1 : nx - 1
        jpp = j + 1 < ny ? j + 1 : 0
        jnn = j - 1 >= 0 ? j - 1 : ny - 1
        joff = is_black ? (isodd(i) ? jpp : jnn) : (isodd(i) ? jnn : jpp)
        nn_sum = op_lattice[inn * ny + j + 1] + op_lattice[i * ny + j + 1] +
                 op_lattice[ipp * ny + j + 1] + op_lattice[i * ny + joff + 1]
        lij = lattice[i * ny + j + 1]
        if randvals[i * ny + j + 1] < exp(-2.0f0 * inv_temp * Float32(nn_sum) * Float32(lij))
            lattice[i * ny + j + 1] = -lij
        end
    end
end

function update_ref!(b, w, randvals, inv_temp, nx, ny)
    update_lattice_ref!(b, w, randvals, inv_temp, nx, ny ÷ 2, true)
    update_lattice_ref!(w, b, randvals, inv_temp, nx, ny ÷ 2, false)
end

function libc_srand(seed::UInt32)
    ccall(:srand, Cvoid, (Cuint,), seed)
end

function libc_rand()
    ccall(:rand, Cint, ())
end

function parse_args()
    nx = 5120
    ny = 5120
    alpha = 0.1f0
    nwarmup = 100
    niters = 1000
    seed = UInt32(1234)
    i = 1
    while i <= length(ARGS)
        opt = ARGS[i]
        if opt == "-x"; i += 1; nx = parse(Int, ARGS[i])
        elseif opt == "-y"; i += 1; ny = parse(Int, ARGS[i])
        elseif opt == "-a"; i += 1; alpha = parse(Float32, ARGS[i])
        elseif opt == "-s"; i += 1; seed = UInt32(parse(UInt64, ARGS[i]))
        elseif opt == "-w"; i += 1; nwarmup = parse(Int, ARGS[i])
        elseif opt == "-n"; i += 1; niters = parse(Int, ARGS[i])
        else
            println("Usage: main.jl -x <rows> -y <cols> -w <warmup> -n <iters>")
            return nothing
        end
        i += 1
    end
    return nx, ny, alpha, nwarmup, niters, seed
end

function main()
    parsed = parse_args()
    parsed === nothing && return 1
    nx, ny, alpha, nwarmup, niters, seed = parsed
    if isodd(nx) || isodd(ny)
        println(stderr, "ERROR: Lattice dimensions must be even values.")
        return 1
    end
    inv_temp = 1.0f0 / (alpha * TCRIT)
    half_cols = ny ÷ 2
    count = nx * half_cols
    libc_srand(seed)
    randvals = Vector{Float32}(undef, count)
    for i in 1:count
        randvals[i] = Float32(libc_rand()) / Float32(typemax(Cint))
    end

    d_rand = CuArray(randvals)
    d_b = CUDA.zeros(Int8, count)
    d_w = CUDA.zeros(Int8, count)
    blocks = cld(count, THREADS)
    @cuda threads=THREADS blocks=blocks init_spins_kernel!(d_b, d_rand, Int64(nx), Int64(half_cols))
    @cuda threads=THREADS blocks=blocks init_spins_kernel!(d_w, d_rand, Int64(nx), Int64(half_cols))

    println("Starting warmup...")
    for _ in 1:nwarmup
        @cuda threads=THREADS blocks=blocks update_lattice_kernel!(d_b, d_w, d_rand, inv_temp, Int64(nx), Int64(half_cols), true)
        @cuda threads=THREADS blocks=blocks update_lattice_kernel!(d_w, d_b, d_rand, inv_temp, Int64(nx), Int64(half_cols), false)
    end
    CUDA.synchronize()

    println("Starting trial iterations...")
    t0 = time()
    for _ in 1:niters
        @cuda threads=THREADS blocks=blocks update_lattice_kernel!(d_b, d_w, d_rand, inv_temp, Int64(nx), Int64(half_cols), true)
        @cuda threads=THREADS blocks=blocks update_lattice_kernel!(d_w, d_b, d_rand, inv_temp, Int64(nx), Int64(half_cols), false)
    end
    CUDA.synchronize()
    duration_us = (time() - t0) * 1.0e6

    println("REPORT:")
    println("\tnGPUs: 1")
    @printf("\ttemperature: %f * %f\n", alpha, TCRIT)
    println("\tseed: $(UInt64(seed))")
    println("\twarmup iterations: $nwarmup")
    println("\ttrial iterations: $niters")
    println("\tlattice dimensions: $nx x $ny")
    @printf("\telapsed time: %f sec\n", duration_us * 1.0e-6)
    @printf("\tupdates per ns: %f\n", Float64(nx * ny) * niters / duration_us * 1.0e-3)

    b_ref = Vector{Int8}(undef, count)
    w_ref = Vector{Int8}(undef, count)
    println("Starting verification iterations ...")
    init_spins_ref!(b_ref, randvals, nx, half_cols)
    init_spins_ref!(w_ref, randvals, nx, half_cols)
    for _ in 1:(nwarmup + niters)
        update_ref!(b_ref, w_ref, randvals, inv_temp, nx, ny)
    end

    b_host = Array(d_b)
    w_host = Array(d_w)
    println((b_host == b_ref && w_host == w_ref) ? "PASS" : "FAIL")
    return 0
end

exit(main())
