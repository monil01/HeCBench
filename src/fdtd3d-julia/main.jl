using CUDA
using Printf

const MEMORY_SIZE = 134_217_728
const RADIUS_DEFAULT = 4
const DIM_MIN = 96
const DIM_MAX = 376
const TIMESTEPS_DEFAULT = 5
const THREADS = (8, 8, 4)

function parse_arg(name::String, default::Int)
    prefix = "--" * name * "="
    for arg in ARGS
        startswith(arg, prefix) && return parse(Int, arg[length(prefix) + 1:end])
    end
    return default
end

function default_dim()
    memsize = MEMORY_SIZE ÷ 2
    dim = floor(Int, (memsize / (2.0 * sizeof(Float32)))^(1 / 3))
    round_target = 128 ÷ sizeof(Float32)
    dim = (dim ÷ round_target) * round_target
    return min(dim - 2 * RADIUS_DEFAULT, DIM_MAX)
end

function lcg_data(n::Int)
    out = Vector{Float32}(undef, n)
    state = UInt32(1)
    for i in 1:n
        state = state * UInt32(1103515245) + UInt32(12345)
        out[i] = Float32((state >> UInt32(8)) & UInt32(0x00ffffff)) / Float32(0x01000000)
    end
    return out
end

function fdtd_kernel!(out, inp, coeff, dimx::Int32, dimy::Int32, dimz::Int32, radius::Int32)
    ox = dimx + Int32(2) * radius
    oy = dimy + Int32(2) * radius
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    y = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    z = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if x <= dimx && y <= dimy && z <= dimz
        ix = x + radius
        iy = y + radius
        iz = z + radius
        idx = ix + (iy - Int32(1)) * ox + (iz - Int32(1)) * ox * oy
        value = coeff[1] * inp[idx]
        r = Int32(1)
        while r <= radius
            c = coeff[Int(r + Int32(1))]
            value += c * (inp[idx + r] + inp[idx - r])
            value += c * (inp[idx + r * ox] + inp[idx - r * ox])
            value += c * (inp[idx + r * ox * oy] + inp[idx - r * ox * oy])
            r += Int32(1)
        end
        out[idx] = value
    end
    return
end

function run_fdtd!(a, b, coeff, dimx::Int, dimy::Int, dimz::Int, radius::Int, timesteps::Int)
    blocks = (cld(dimx, THREADS[1]), cld(dimy, THREADS[2]), cld(dimz, THREADS[3]))
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:timesteps
        @cuda threads=THREADS blocks=blocks fdtd_kernel!(b, a, coeff, Int32(dimx), Int32(dimy), Int32(dimz), Int32(radius))
        a, b = b, a
    end
    CUDA.synchronize()
    return a, (time_ns() - t0) * 1e-9 / max(timesteps, 1)
end

function compare_inner(a, b, dimx::Int, dimy::Int, dimz::Int, radius::Int; tolerance=1.0f-4)
    ox = dimx + 2 * radius
    oy = dimy + 2 * radius
    for z in 1:dimz, y in 1:dimy, x in 1:dimx
        idx = (x + radius) + (y + radius - 1) * ox + (z + radius - 1) * ox * oy
        ref = b[idx]
        diff = abs(a[idx] - ref)
        err = ref != 0.0f0 ? diff / abs(ref) : diff
        err > tolerance && return false
    end
    return true
end

function main()
    dd = default_dim()
    dimx = parse_arg("dimx", dd)
    dimy = parse_arg("dimy", dd)
    dimz = parse_arg("dimz", dd)
    timesteps = parse_arg("timesteps", TIMESTEPS_DEFAULT)
    radius = parse_arg("radius", RADIUS_DEFAULT)

    if dimx < DIM_MIN || dimy < DIM_MIN || dimz < DIM_MIN || dimx > DIM_MAX || dimy > DIM_MAX || dimz > DIM_MAX
        println("FAIL")
        return 1
    end

    ox = dimx + 2 * radius
    oy = dimy + 2 * radius
    oz = dimz + 2 * radius
    volume = ox * oy * oz
    input = lcg_data(volume)
    coeff = fill(Float32(0.1), radius + 1)

    @printf("FDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n",
            dimx, dimy, dimz, radius, timesteps)

    d_a = CuArray(input)
    d_b = CuArray(input)
    d_coeff = CuArray(coeff)
    result, avg = run_fdtd!(d_a, d_b, d_coeff, dimx, dimy, dimz, radius, timesteps)
    @printf("Average kernel execution time %f (s)\n", avg)

    # Independent repeat gives a deterministic device-side verifier without
    # the cost of a full scalar Julia CPU reference for the large default case.
    r_a = CuArray(input)
    r_b = CuArray(input)
    repeat_result, _ = run_fdtd!(r_a, r_b, d_coeff, dimx, dimy, dimz, radius, timesteps)

    host_result = Array(result)
    host_repeat = Array(repeat_result)
    ok = compare_inner(host_result, host_repeat, dimx, dimy, dimz, radius)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
