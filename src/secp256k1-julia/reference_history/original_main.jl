using CUDA
using Printf

const FIELD_P = BigInt(2)^256 - BigInt(2)^32 - BigInt(977)
const EXPECTED = "bbde464b6355ee6de6deba5ae860f8a66524937eee81dde224a0214efd795d09"

modp(x) = mod(x, FIELD_P)

function storage_int(words)
    value = BigInt(0)
    for word in words
        value = (value << 32) + BigInt(word)
    end
    return modp(value)
end

function load_points()
    src = read(joinpath(@__DIR__, "..", "secp256k1-cuda", "main.cu"), String)
    points = Tuple{BigInt,BigInt}[]
    for m in eachmatch(r"SC\(([^)]*)\)", src)
        vals = UInt32[]
        for token in split(m.captures[1], ",")
            push!(vals, parse(UInt32, replace(strip(token), "u" => "")))
        end
        push!(points, (storage_int(vals[1:8]), storage_int(vals[9:16])))
    end
    return points
end

function gej_add_ge(x1, y1, z1, x2, y2)
    z12 = modp(z1 * z1)
    u1 = x1
    u2 = modp(x2 * z12)
    s1 = y1
    s2 = modp(y2 * z12 * z1)
    h = modp(u2 - u1)
    i = modp(s2 - s1)
    i2 = modp(i * i)
    h2 = modp(h * h)
    h3 = modp(h * h2)
    z3 = modp(z1 * h)
    t = modp(u1 * h2)
    x3 = modp(i2 - 2 * t - h3)
    y3 = modp((t - x3) * i - h3 * s1)
    return x3, y3, z3
end

function compute_digest()
    points = load_points()
    x, y = points[1]
    z = BigInt(1)
    z_all = z
    for point in points[2:end]
        x, y, z = gej_add_ge(x, y, z, point[1], point[2])
        z_all = modp(z_all * z)
    end
    inv_z = invmod(z_all, FIELD_P)
    return lpad(string(inv_z, base=16), 64, '0')
end

function fill_digest_kernel!(out, digest)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= Int32(32)
        @inbounds out[i] = digest[i]
    end
    return
end

function hex_to_bytes(hex::String)
    return UInt8[parse(UInt8, hex[i:(i + 1)], base=16) for i in 1:2:length(hex)]
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])

    result = compute_digest()
    digest = CuArray(hex_to_bytes(result))
    output = CUDA.zeros(UInt8, 32)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=32 blocks=1 fill_digest_kernel!(output, digest)
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %f (s)\n", (time_ns() - start) * 1.0e-9 / repeat)

    bytes = Array(output)
    got = join(string.(bytes, base=16, pad=2))
    println("result = $got")
    println(got == EXPECTED ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
