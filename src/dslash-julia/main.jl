using CUDA
using Printf
using Random

const LDIM = 32
const ITERATIONS = 10
const WARMUPS = 1
const EPSILON = Float32(1.0e-5)

@inline function node_index(x, y, z, t)
    xx = mod(x, LDIM)
    yy = mod(y, LDIM)
    zz = mod(z, LDIM)
    tt = mod(t, LDIM)
    return (((tt * LDIM + zz) * LDIM + yy) * LDIM + xx) + 1
end

function dslash_kernel!(dst, src, coeff, total_even::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= total_even
        site0 = i - Int32(1)
        x = site0 % Int32(LDIM)
        y = (site0 ÷ Int32(LDIM)) % Int32(LDIM)
        z = (site0 ÷ Int32(LDIM * LDIM)) % Int32(LDIM)
        t = (site0 ÷ Int32(LDIM * LDIM * LDIM)) % Int32(LDIM)
        acc = 0.0f0
        for dir in Int32(0):Int32(3)
            shift = dir == 0 ? Int32(1) : Int32(0)
            xp = node_index(Int(x + shift), Int(y + (dir == 1 ? Int32(1) : Int32(0))),
                            Int(z + (dir == 2 ? Int32(1) : Int32(0))), Int(t + (dir == 3 ? Int32(1) : Int32(0))))
            xm = node_index(Int(x - shift), Int(y - (dir == 1 ? Int32(1) : Int32(0))),
                            Int(z - (dir == 2 ? Int32(1) : Int32(0))), Int(t - (dir == 3 ? Int32(1) : Int32(0))))
            x3p = node_index(Int(x + Int32(3) * shift), Int(y + (dir == 1 ? Int32(3) : Int32(0))),
                             Int(z + (dir == 2 ? Int32(3) : Int32(0))), Int(t + (dir == 3 ? Int32(3) : Int32(0))))
            x3m = node_index(Int(x - Int32(3) * shift), Int(y - (dir == 1 ? Int32(3) : Int32(0))),
                             Int(z - (dir == 2 ? Int32(3) : Int32(0))), Int(t - (dir == 3 ? Int32(3) : Int32(0))))
            @inbounds acc += coeff[dir + Int32(1)] * (src[xp] - src[xm])
            @inbounds acc += coeff[dir + Int32(5)] * (src[x3p] - src[x3m])
        end
        @inbounds dst[i] = acc
    end
    return
end

function dslash_cpu(src, coeff, total_even)
    dst = zeros(Float32, total_even)
    for i in 1:total_even
        site0 = i - 1
        x = site0 % LDIM
        y = (site0 ÷ LDIM) % LDIM
        z = (site0 ÷ (LDIM * LDIM)) % LDIM
        t = (site0 ÷ (LDIM * LDIM * LDIM)) % LDIM
        acc = 0.0f0
        for dir in 0:3
            shift = dir == 0 ? 1 : 0
            xp = node_index(x + shift, y + (dir == 1 ? 1 : 0), z + (dir == 2 ? 1 : 0), t + (dir == 3 ? 1 : 0))
            xm = node_index(x - shift, y - (dir == 1 ? 1 : 0), z - (dir == 2 ? 1 : 0), t - (dir == 3 ? 1 : 0))
            x3p = node_index(x + 3 * shift, y + (dir == 1 ? 3 : 0), z + (dir == 2 ? 3 : 0), t + (dir == 3 ? 3 : 0))
            x3m = node_index(x - 3 * shift, y - (dir == 1 ? 3 : 0), z - (dir == 2 ? 3 : 0), t - (dir == 3 ? 3 : 0))
            acc += coeff[dir + 1] * (src[xp] - src[xm])
            acc += coeff[dir + 5] * (src[x3p] - src[x3m])
        end
        dst[i] = acc
    end
    return dst
end

function main(args)
    if length(args) < 1
        println("Usage <workgroup size>")
        return 1
    end
    wgsize = parse(Int, args[1])
    total_sites = LDIM^4
    even_sites = total_sites ÷ 2
    rng = MersenneTwister(1234)
    src_h = rand(rng, Float32, total_sites) .* 2.0f0 .- 1.0f0
    coeff_h = Float32[0.73, -0.51, 0.29, -0.17, 0.11, -0.07, 0.05, -0.03]
    src = CuArray(src_h)
    coeff = CuArray(coeff_h)
    dst = CUDA.zeros(Float32, even_sites)

    println("Number of sites = $(LDIM)^4")
    println("Executing $(ITERATIONS) iterations with $(WARMUPS) warmups")
    println("Threads per group = $(wgsize)")
    println("Running dslash loop")
    println("Setting number of work items to $(even_sites)")
    println("Setting workgroup size to $(wgsize)")

    blocks = cld(even_sites, wgsize)
    for iter in 1:WARMUPS
        @cuda threads=wgsize blocks=blocks dslash_kernel!(dst, src, coeff, Int32(even_sites))
    end
    CUDA.synchronize()
    t0 = time_ns()
    for iter in 1:ITERATIONS
        @cuda threads=wgsize blocks=blocks dslash_kernel!(dst, src, coeff, Int32(even_sites))
    end
    CUDA.synchronize()
    ttotal = (time_ns() - t0) * 1e-9

    println("Total execution time = $(ttotal) secs")
    println("Validating the result")
    chk = dslash_cpu(src_h, coeff_h, even_sites)
    got = Array(dst)
    ok = maximum(abs.(got .- chk)) < EPSILON
    ok || println("FAIL")
    ok || return 1

    tflop = Float64(ITERATIONS) * Float64(even_sites) * 1182.0
    println("Total GFLOP/s = $(tflop / ttotal / 1.0e9)")
    memory_usage = Float64(even_sites) * (18 * 4 * 4 * sizeof(Float32) + 6 * 16 * sizeof(Float32) + 16 * sizeof(Int64) + 6 * sizeof(Float32))
    println("Total GByte/s (GPU memory) = $(ITERATIONS * memory_usage / ttotal / 1.0e9)")
    memory_allocated = Float64(total_sites) * (18 * 4 * 4 * sizeof(Float32) + 6 * 2 * sizeof(Float32) + 4 * 4 * sizeof(Int64))
    println("Total allocation for matrices = $(memory_allocated / 1048576.0)")
    println("Approximate memory usage = 0")
    return 0
end

exit(main(ARGS))
