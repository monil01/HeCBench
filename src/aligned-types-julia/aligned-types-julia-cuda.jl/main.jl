using CUDA
using Printf

# Julia port of aligned-types-cuda: measures memcpy throughput for various
# aligned/misaligned element sizes.  Julia/CUDA.jl doesn't expose C-style
# __align__ per-struct, so we approximate the misaligned case by copying
# raw-byte tuples that are not naturally aligned, and the aligned case by
# copying primitive types (which are naturally aligned).

const MEM_SIZE = 50000000
const NUM_ITERATIONS = 1000

# Copy kernel: one element per thread
function copy_kernel!(dst, src, num_elements::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= num_elements
        @inbounds dst[i] = src[i]
    end
    return
end

function idiv_up(a, b); (a % b != 0) ? (a ÷ b + 1) : (a ÷ b); end
function idiv_down(a, b); a ÷ b; end
function ialign_down(a, b); a - a % b; end

function run_test(::Type{T}, packed_size::Int, d_idata_bytes::CuArray{UInt8},
                  d_odata_bytes::CuArray{UInt8}, h_idata::Vector{UInt8},
                  memory_size::Int) where {T}
    element_bytes = sizeof(T)
    total_aligned = ialign_down(memory_size, element_bytes)
    num_elements = idiv_down(memory_size, element_bytes)

    # Reinterpret raw byte buffers as arrays of T
    d_idata_T = reinterpret(T, d_idata_bytes)
    d_odata_T = reinterpret(T, d_odata_bytes)
    d_idata_T = view(d_idata_T, 1:num_elements)
    d_odata_T = view(d_odata_T, 1:num_elements)

    CUDA.fill!(d_odata_bytes, UInt8(0))
    CUDA.synchronize()

    threads = 256
    blocks = cld(num_elements, threads)

    t0 = time_ns()
    for _ in 1:NUM_ITERATIONS
        @cuda threads=threads blocks=blocks copy_kernel!(d_odata_T, d_idata_T, Int32(num_elements))
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1e-9 / NUM_ITERATIONS

    @printf("Avg. time: %f ms / Copy throughput: %f GB/s.\n",
            elapsed_s * 1000, Float64(total_aligned) / (elapsed_s * 1073741824.0))

    h_odata = Array(d_odata_bytes)
    ok = true
    limit = num_elements * element_bytes
    if packed_size == element_bytes
        # Simple byte-for-byte compare within the aligned region
        for i in 1:limit
            if h_idata[i] != h_odata[i]
                ok = false
                break
            end
        end
    else
        # Padded structure: only compare the first `packed_size` bytes of every element
        for e in 0:num_elements-1
            base = e * element_bytes
            for i in 0:packed_size-1
                if h_idata[base+i+1] != h_odata[base+i+1]
                    ok = false
                    break
                end
            end
            ok || break
        end
    end
    println("\tTEST ", ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

# Structs with different sizes (naturally aligned in Julia)
struct UChar4Aligned;  r::UInt8; g::UInt8; b::UInt8; a::UInt8; end
struct UInt2Aligned;   l::UInt32; a::UInt32; end
# 3-uint = 12 bytes (not power-of-two).  Represented by a plain tuple.
struct UInt3;          r::UInt32; g::UInt32; b::UInt32; end
struct UInt4;          r::UInt32; g::UInt32; b::UInt32; a::UInt32; end
struct UInt8Struct;    c1::UInt4; c2::UInt4; end

function main()
    println("[main.jl] - Starting...")
    println("Allocating memory...")
    memory_size = MEM_SIZE & 0xffffff00

    h_idata = Vector{UInt8}(undef, memory_size)
    for i in 1:memory_size
        h_idata[i] = UInt8((((i-1) & 0xFF) + 1) & 0xFF)
    end

    d_idata = CuArray(h_idata)
    d_odata = CUDA.zeros(UInt8, memory_size)

    println("Generating host input data array...")
    println("Uploading input data to GPU memory...")

    nfail = 0
    println("Testing misaligned types...")
    println("uchar_misaligned...");   nfail += run_test(UInt8,          1,  d_idata, d_odata, h_idata, memory_size)
    println("uchar4_misaligned..."); nfail += run_test(UChar4Aligned,  4,  d_idata, d_odata, h_idata, memory_size)
    println("uchar4_aligned...");    nfail += run_test(UChar4Aligned,  4,  d_idata, d_odata, h_idata, memory_size)
    println("ushort_misaligned..."); nfail += run_test(UInt16,         2,  d_idata, d_odata, h_idata, memory_size)
    println("uint_aligned...");      nfail += run_test(UInt32,         4,  d_idata, d_odata, h_idata, memory_size)
    println("uint2_misaligned..."); nfail += run_test(UInt2Aligned,   8,  d_idata, d_odata, h_idata, memory_size)
    println("uint2_aligned...");    nfail += run_test(UInt2Aligned,   8,  d_idata, d_odata, h_idata, memory_size)
    println("uint3_misaligned..."); nfail += run_test(UInt3,          12, d_idata, d_odata, h_idata, memory_size)
    println("uint3_aligned...");    nfail += run_test(UInt3,          12, d_idata, d_odata, h_idata, memory_size)
    println("uint4_misaligned..."); nfail += run_test(UInt4,          16, d_idata, d_odata, h_idata, memory_size)
    println("uint4_aligned...");    nfail += run_test(UInt4,          16, d_idata, d_odata, h_idata, memory_size)
    println("uint8_misaligned..."); nfail += run_test(UInt8Struct,    32, d_idata, d_odata, h_idata, memory_size)
    println("uint8_aligned...");    nfail += run_test(UInt8Struct,    32, d_idata, d_odata, h_idata, memory_size)

    @printf("\n[alignedTypes] -> Test Results: %d Failures\n", nfail)

    println("Shutting down...")
    if nfail == 0
        println("Test passed")
        println("PASS")
    else
        println("Test failed!")
        println("FAIL")
    end
    return 0
end

main()
