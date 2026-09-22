using CUDA
using Printf

const FACTOR = Int32(100)
const NTAB = Int32(4)
const IA = Int32(16807)
const IM = Int32(2147483647)
const AM = Float32(1.0 / 2147483647.0)
const IQ = Int32(127773)
const IR = Int32(2836)
const NDIV = Int32(1 + (IM - Int32(1)) ÷ NTAB)

function read_u16_le(bytes, offset)
    return UInt16(bytes[offset + 1]) | (UInt16(bytes[offset + 2]) << 8)
end

function read_u32_le(bytes, offset)
    return UInt32(bytes[offset + 1]) | (UInt32(bytes[offset + 2]) << 8) |
           (UInt32(bytes[offset + 3]) << 16) | (UInt32(bytes[offset + 4]) << 24)
end

function read_i32_le(bytes, offset)
    return reinterpret(Int32, read_u32_le(bytes, offset))
end

function resolve_input(path)
    isfile(path) && return path
    fallback = joinpath(@__DIR__, "..", "urng-sycl", basename(path))
    return isfile(fallback) ? fallback : path
end

function load_bmp_rgba(path)
    bytes = read(resolve_input(path))
    length(bytes) >= 54 || error("BMP header too short")
    bytes[1] == UInt8('B') && bytes[2] == UInt8('M') || error("not a BMP file")
    offset = Int(read_u32_le(bytes, 10))
    width = Int(read_i32_le(bytes, 18))
    raw_height = Int(read_i32_le(bytes, 22))
    bit_count = Int(read_u16_le(bytes, 28))
    compression = Int(read_u32_le(bytes, 30))
    bit_count == 24 || bit_count == 32 || error("only 24/32-bit BMP supported")
    compression == 0 || error("compressed BMP not supported")
    height = abs(raw_height)
    top_down = raw_height < 0
    row_stride = ((width * bit_count + 31) ÷ 32) * 4
    pixels = Matrix{NTuple{4,UInt8}}(undef, height, width)
    channels = bit_count ÷ 8
    for y in 0:(height - 1)
        src_y = top_down ? y : (height - 1 - y)
        row = offset + src_y * row_stride
        for x in 0:(width - 1)
            pos = row + x * channels
            b = bytes[pos + 1]
            g = bytes[pos + 2]
            r = bytes[pos + 3]
            a = channels == 4 ? bytes[pos + 4] : UInt8(255)
            pixels[y + 1, x + 1] = (r, g, b, a)
        end
    end
    return pixels, width, height
end

function ran1_device(idum::Int32, iv)
    iy = Int32(0)
    tid0 = threadIdx().x - Int32(1)
    for j in NTAB:-Int32(0)
        k = idum ÷ IQ
        idum = IA * (idum - k * IQ) - IR * k
        if idum < 0
            idum += IM
        end
        if j < NTAB
            iv[Int(j) * blockDim().x + Int(tid0) + 1] = idum
        end
    end
    iy = iv[Int(tid0) + 1]
    k = idum ÷ IQ
    idum = IA * (idum - k * IQ) - IR * k
    if idum < 0
        idum += IM
    end
    j = iy ÷ NDIV
    iy = iv[Int(j) * blockDim().x + Int(tid0) + 1]
    return AM * Float32(iy)
end

function clamp_u8(x::Float32)
    y = ifelse(x > 255f0, 255f0, ifelse(x < 0f0, 0f0, x))
    return UInt8(trunc(Int32, y))
end

function noise_uniform_kernel!(input, output, size::Int32, factor::Int32)
    pos0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if pos0 < size
        idx = Int(pos0) + 1
        pix = input[idx]
        x = Float32(pix[1])
        y = Float32(pix[2])
        z = Float32(pix[3])
        w = Float32(pix[4])
        avg = (x + y + z + w) / 4f0
        iv = @cuDynamicSharedMem(Int32, Int(NTAB) * Int(blockDim().x))
        dev = ran1_device(-trunc(Int32, avg), iv)
        dev = (dev - 0.55f0) * Float32(factor)
        output[idx] = (clamp_u8(x + dev), clamp_u8(y + dev), clamp_u8(z + dev), clamp_u8(w + dev))
    end
    return
end

function main()
    length(ARGS) == 2 || begin
        println("Usage: main.jl <path to file> <repeat>")
        exit(1)
    end
    file_path = ARGS[1]
    iterations = parse(Int, ARGS[2])
    iterations > 0 || error("repeat must be positive")
    CUDA.allowscalar(false)

    pixels, width, height = load_bmp_rgba(file_path)
    input = vec(pixels)
    size = length(input)
    println("Image $file_path height: $height width: $width")

    d_input = CuArray(input)
    d_output = CuArray(fill((UInt8(0), UInt8(0), UInt8(0), UInt8(0)), size))
    threads = 256
    blocks = cld(size, threads)
    shmem = Int(NTAB) * threads * sizeof(Int32)

    println("Executing kernel for $iterations iterations")
    println("-------------------------------------------")
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:iterations
        @cuda threads=threads blocks=blocks shmem=shmem noise_uniform_kernel!(d_input, d_output, Int32(size), FACTOR)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time: %f (us)\n", elapsed * 1.0e-3 / iterations)

    output = Array(d_output)
    total = 0.0
    @inbounds for i in eachindex(input)
        total += Int(output[i][1]) - Int(input[i][1])
        total += Int(output[i][2]) - Int(input[i][2])
        total += Int(output[i][3]) - Int(input[i][3])
        total += Int(output[i][4]) - Int(input[i][4])
    end
    mean = total / (size * 4 * Int(FACTOR))
    println("The averaged mean of the image: $mean")
    println(abs(mean) < 1.0 ? "PASS" : "FAIL")
end

main()
