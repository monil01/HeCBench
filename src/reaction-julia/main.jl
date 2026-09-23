using CUDA
using Printf
using Random

function build_input_central_cube(ncells, mx, my, mz, a, b, a0, b0, ca, cb, delta)
    rng = MersenneTwister(123)
    for i in 1:ncells
        a[i] = a0 + rand(rng, Float32) * delta
        b[i] = b0 + rand(rng, Float32) * delta
    end

    cbsz = 5
    for z in (mz ÷ 2 - cbsz):(mz ÷ 2 + cbsz - 1)
        for y in (my ÷ 2 - cbsz):(my ÷ 2 + cbsz - 1)
            for x in (mx ÷ 2 - cbsz):(mx ÷ 2 + cbsz - 1)
                idx = z * mx * my + y * mx + x + 1
                a[idx] = ca + rand(rng, Float32) * delta
                b[idx] = cb + rand(rng, Float32) * delta
            end
        end
    end
end

function reaction_step_zeroflux!(a, b, a_next, b_next,
                                 ncells::Int32, mx::Int32, my::Int32, mz::Int32,
                                 diffcon_a::Float32, diffcon_b::Float32,
                                 c1::Float32, c2::Float32, dt::Float32)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    plane = mx * my

    while index < ncells
        x = index % mx
        y = (index ÷ mx) % my
        z = index ÷ plane

        center_a = @inbounds a[index + Int32(1)]
        center_b = @inbounds b[index + Int32(1)]

        xm = x == Int32(0) ? center_a : @inbounds a[index]
        xp = x == mx - Int32(1) ? center_a : @inbounds a[index + Int32(2)]
        ym = y == Int32(0) ? center_a : @inbounds a[index - mx + Int32(1)]
        yp = y == my - Int32(1) ? center_a : @inbounds a[index + mx + Int32(1)]
        zm = z == Int32(0) ? center_a : @inbounds a[index - plane + Int32(1)]
        zp = z == mz - Int32(1) ? center_a : @inbounds a[index + plane + Int32(1)]
        lap_a = (xm - center_a) + (xp - center_a) +
                (ym - center_a) + (yp - center_a) +
                (zm - center_a) + (zp - center_a)

        xm_b = x == Int32(0) ? center_b : @inbounds b[index]
        xp_b = x == mx - Int32(1) ? center_b : @inbounds b[index + Int32(2)]
        ym_b = y == Int32(0) ? center_b : @inbounds b[index - mx + Int32(1)]
        yp_b = y == my - Int32(1) ? center_b : @inbounds b[index + mx + Int32(1)]
        zm_b = z == Int32(0) ? center_b : @inbounds b[index - plane + Int32(1)]
        zp_b = z == mz - Int32(1) ? center_b : @inbounds b[index + plane + Int32(1)]
        lap_b = (xm_b - center_b) + (xp_b - center_b) +
                (ym_b - center_b) + (yp_b - center_b) +
                (zm_b - center_b) + (zp_b - center_b)

        r = center_a * center_b * center_b
        ra = -r + c1 * (1.0f0 - center_a)
        rb = r - (c1 + c2) * center_b

        @inbounds a_next[index + Int32(1)] = center_a + (diffcon_a * lap_a + ra) * dt
        @inbounds b_next[index + Int32(1)] = center_b + (diffcon_b * lap_b + rb) * dt
        index += stride
    end
    return
end

function stats(a, b)
    min_a = minimum(a)
    min_b = minimum(b)
    max_a = maximum(a)
    max_b = maximum(b)
    println("  Components A | B ")
    @printf("  Min = %12.6f | %12.6f\n", min_a, min_b)
    @printf("  Max = %12.6f | %12.6f\n", max_a, max_b)
end

function main()
    if length(ARGS) != 1
        @printf("Usage: %s <timesteps>\n", PROGRAM_FILE)
        exit(1)
    end
    timesteps = parse(Int, ARGS[1])

    mx = 128
    my = 128
    mz = 128
    ncells = mx * my * mz

    da = 0.16f0
    db = 0.08f0
    dt = 0.25f0
    dx = 0.5f0
    c1 = 0.0392f0
    c2 = 0.0649f0

    println("Starting time-integration")
    println("Constructing initial concentrations...")
    a = Vector{Float32}(undef, ncells)
    b = Vector{Float32}(undef, ncells)
    build_input_central_cube(ncells, mx, my, mz, a, b, 1.0f0, 0.0f0, 0.5f0, 0.25f0, 0.05f0)

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_a_next = similar(d_a)
    d_b_next = similar(d_b)

    diffcon_a = da / (dx * dx)
    diffcon_b = db / (dx * dx)
    threads = 256
    blocks = cld(ncells, threads)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:timesteps
        @cuda threads=threads blocks=blocks reaction_step_zeroflux!(
            d_a, d_b, d_a_next, d_b_next,
            Int32(ncells), Int32(mx), Int32(my), Int32(mz),
            diffcon_a, diffcon_b, c1, c2, dt)
        d_a, d_a_next = d_a_next, d_a
        d_b, d_b_next = d_b_next, d_b
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1e-9

    @printf("timesteps: %d\n", timesteps)
    @printf("Total kernel execution time:     %12.3f s\n\n", elapsed_s)

    stats(Array(d_a), Array(d_b))
end

main()
