using CUDA
using Printf

const CUTSQ = Float32(13.5)
const MAX_NEIGHBORS = 128
const DOMAIN_EDGE = 20
const LJ1 = Float32(1.5)
const LJ2 = Float32(2.0)
const PROB_SIZES = (12288, 24576, 36864, 73728)
const RAND_MAX_F32 = Float32(typemax(Cint))

function libc_srand(seed::Integer)
    ccall(:srand, Cvoid, (Cuint,), Cuint(seed))
end

function libc_rand()
    return ccall(:rand, Cint, ())
end

function md_kernel!(px, py, pz, fx, fy, fz, neighbor_list, n_atom::Int32,
                    max_neighbors::Int32, lj1::Float32, lj2::Float32, cutsq::Float32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if idx0 >= n_atom
        return
    end

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
    return
end

@inline function distance2(px, py, pz, i::Int, j::Int)
    dx = px[i] - px[j]
    dy = py[i] - py[j]
    dz = pz[i] - pz[j]
    return dx * dx + dy * dy + dz * dz
end

function build_neighbor_list(px, py, pz)
    n_atom = length(px)
    neighbor_list = Vector{Int32}(undef, MAX_NEIGHBORS * n_atom)
    total_pairs = 0
    dists = Vector{Float32}(undef, n_atom - 1)
    ids = Vector{Int32}(undef, n_atom - 1)

    for i in 1:n_atom
        k = 1
        for j in 1:n_atom
            if i != j
                dists[k] = distance2(px, py, pz, i, j)
                ids[k] = Int32(j - 1)
                k += 1
            end
        end
        order = partialsortperm(dists, 1:MAX_NEIGHBORS)
        for out_idx in 1:MAX_NEIGHBORS
            src_idx = order[out_idx]
            neighbor_list[(out_idx - 1) * n_atom + i] = ids[src_idx]
            if dists[src_idx] < CUTSQ
                total_pairs += 1
            end
        end
    end
    return neighbor_list, total_pairs
end

function cpu_forces(px, py, pz, neighbor_list)
    n_atom = length(px)
    fx = zeros(Float32, n_atom)
    fy = zeros(Float32, n_atom)
    fz = zeros(Float32, n_atom)
    for i in 1:n_atom
        ipx = px[i]
        ipy = py[i]
        ipz = pz[i]
        ftx = 0.0f0
        fty = 0.0f0
        ftz = 0.0f0
        for j in 0:(MAX_NEIGHBORS - 1)
            jidx = Int(neighbor_list[j * n_atom + i]) + 1
            delx = ipx - px[jidx]
            dely = ipy - py[jidx]
            delz = ipz - pz[jidx]
            r2inv = delx * delx + dely * dely + delz * delz
            if r2inv > 0.0f0 && r2inv < CUTSQ
                r2inv = 1.0f0 / r2inv
                r6inv = r2inv * r2inv * r2inv
                force_c = r2inv * r6inv * (LJ1 * r6inv - LJ2)
                ftx += delx * force_c
                fty += dely * force_c
                ftz += delz * force_c
            end
        end
        fx[i] = ftx
        fy[i] = fty
        fz[i] = ftz
    end
    return fx, fy, fz
end

function run_md!(dpx, dpy, dpz, dfx, dfy, dfz, dneighbor, n_atom::Int, iterations::Int)
    threads = 256
    blocks = cld(n_atom, threads)
    for _ in 1:iterations
        @cuda threads=threads blocks=blocks md_kernel!(
            dpx, dpy, dpz, dfx, dfy, dfz, dneighbor,
            Int32(n_atom), Int32(MAX_NEIGHBORS), LJ1, LJ2, CUTSQ)
    end
    CUDA.synchronize()
end

function check_results(fx_ref, fy_ref, fz_ref, fx, fy, fz)
    max_error = 0.0f0
    for i in eachindex(fx_ref)
        if isnan(fx[i]) || isnan(fy[i]) || isnan(fz[i])
            println("FAIL")
            return false
        end
        max_error = max(max_error, abs(fx_ref[i] - fx[i]), abs(fy_ref[i] - fy[i]), abs(fz_ref[i] - fz[i]))
    end
    @printf("Max error between host and device: %.9g\n", max_error)
    if max_error <= 1.0f-5
        println("PASS")
        return true
    end
    println("FAIL")
    return false
end

function main(args)
    if length(args) != 2
        println("Usage: ./main.jl <class size> <iteration>")
        return 1
    end
    size_class = parse(Int, args[1])
    iteration = parse(Int, args[2])
    if size_class < 0 || size_class >= length(PROB_SIZES) || iteration < 0
        println("Usage: ./main.jl <class size 0-3> <iteration >= 0>")
        return 1
    end
    n_atom = PROB_SIZES[size_class + 1]

    println("Initializing test problem (this can take several minutes for large problems).")
    libc_srand(123)
    px = Vector{Float32}(undef, n_atom)
    py = Vector{Float32}(undef, n_atom)
    pz = Vector{Float32}(undef, n_atom)
    for i in 1:n_atom
        px[i] = Float32(mod(libc_rand(), DOMAIN_EDGE))
        py[i] = Float32(mod(libc_rand(), DOMAIN_EDGE))
        pz[i] = Float32(mod(libc_rand(), DOMAIN_EDGE))
    end
    println("Finished.")

    neighbor_list, total_pairs = build_neighbor_list(px, py, pz)
    @printf("%d of %d pairs within cutoff distance = %.6f %%\n",
            total_pairs, n_atom * MAX_NEIGHBORS,
            100.0 * Float64(total_pairs) / Float64(n_atom * MAX_NEIGHBORS))

    dpx = CuArray(px)
    dpy = CuArray(py)
    dpz = CuArray(pz)
    dfx = CUDA.zeros(Float32, n_atom)
    dfy = CUDA.zeros(Float32, n_atom)
    dfz = CUDA.zeros(Float32, n_atom)
    dneighbor = CuArray(neighbor_list)

    run_md!(dpx, dpy, dpz, dfx, dfy, dfz, dneighbor, n_atom, 1)
    println("Performing Correctness Check (may take several minutes)")
    fx_ref, fy_ref, fz_ref = cpu_forces(px, py, pz, neighbor_list)
    ok = check_results(fx_ref, fy_ref, fz_ref, Array(dfx), Array(dfy), Array(dfz))
    ok || return 1

    CUDA.synchronize()
    start = time_ns()
    run_md!(dpx, dpy, dpz, dfx, dfy, dfz, dneighbor, n_atom, iteration)
    elapsed = (time_ns() - start) * 1.0e-9
    @printf("Average kernel execution time %.9g (s)\n", elapsed / iteration)
    return 0
end

exit(main(ARGS))
