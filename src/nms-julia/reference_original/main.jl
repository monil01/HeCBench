using CUDA
using Printf

const MAX_DETECTIONS = Int32(4096)
const N_PARTITIONS = Int32(32)
const OTHRESHOLD = Float32(0.3)

function generate_nms_bitmap_kernel!(xs, ys, ws, scores, bitmap, limit::Int32)
    i0 = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    j0 = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    if i0 < limit && j0 < limit
        i = i0 + Int32(1)
        j = j0 + Int32(1)
        if scores[i] < scores[j]
            area = (ws[j] + Float32(1)) * (ws[j] + Float32(1))
            iw = max(Float32(0), min(xs[i] + ws[i], xs[j] + ws[j]) - max(xs[i], xs[j]) + Float32(1))
            ih = max(Float32(0), min(ys[i] + ws[i], ys[j] + ws[j]) - max(ys[i], ys[j]) + Float32(1))
            keep = (((iw * ih) / area) < OTHRESHOLD) && (ws[j] != Float32(0))
            bitmap[Int(i0) * Int(MAX_DETECTIONS) + Int(j0) + 1] = keep ? UInt8(1) : UInt8(0)
        end
    end
    return
end

function reduce_nms_bitmap_kernel!(bitmap, pointsbitmap, ndetections::Int32)
    row0 = Int32(blockIdx().x - Int32(1))
    tx0 = Int32(threadIdx().x - Int32(1))
    if row0 < ndetections
        keep = UInt8(1)
        stride = MAX_DETECTIONS ÷ N_PARTITIONS
        for part0 in Int32(0):(N_PARTITIONS - Int32(1))
            col0 = tx0 + part0 * stride
            keep &= bitmap[Int(row0) * Int(MAX_DETECTIONS) + Int(col0) + 1]
        end
        shared = @cuStaticSharedMem(UInt8, 128)
        shared[threadIdx().x] = keep
        sync_threads()
        step = Int32(64)
        while step > 0
            if tx0 < step
                shared[threadIdx().x] &= shared[threadIdx().x + step]
            end
            sync_threads()
            step ÷= Int32(2)
        end
        if tx0 == 0
            pointsbitmap[Int(row0) + 1] = shared[1]
        end
    end
    return
end

function resolve_input(path::String)
    if isfile(path)
        return path
    end
    fallback = joinpath(@__DIR__, "..", "nms-cuda", basename(path))
    return isfile(fallback) ? fallback : path
end

function read_detections(path::String)
    xs = zeros(Float32, Int(MAX_DETECTIONS))
    ys = zeros(Float32, Int(MAX_DETECTIONS))
    ws = zeros(Float32, Int(MAX_DETECTIONS))
    scores = zeros(Float32, Int(MAX_DETECTIONS))
    ndetections = 0
    open(resolve_input(path), "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, ',')
            length(parts) == 4 || error("invalid detection line $(ndetections + 1)")
            ndetections += 1
            ndetections <= Int(MAX_DETECTIONS) || error("too many detections")
            xs[ndetections] = Float32(parse(Int32, parts[1]))
            ys[ndetections] = Float32(parse(Int32, parts[2]))
            ws[ndetections] = Float32(parse(Int32, parts[3]))
            scores[ndetections] = parse(Float32, parts[4])
        end
    end
    return xs, ys, ws, scores, ndetections
end

function get_optimal_dim(val::Int32)
    div = Int32(16)
    cntneg = div
    cntpos = div
    neg = true
    for _ in 1:5
        val % div == 0 && return div
        if neg
            cntneg -= Int32(1)
            div = cntneg
            neg = false
        else
            cntpos += Int32(1)
            div = cntpos
            neg = true
        end
    end
    return Int32(16)
end

function get_upper_limit(val::Int32, mul::Int32)
    cnt = mul
    while cnt < val
        cnt += mul
    end
    return min(cnt, MAX_DETECTIONS)
end

function cpu_nms(xs, ys, ws, scores, ndetections::Int)
    keep = trues(ndetections)
    for i in 1:ndetections
        for j in 1:Int(MAX_DETECTIONS)
            value = true
            if scores[i] < scores[j]
                area = (ws[j] + 1f0) * (ws[j] + 1f0)
                iw = max(0f0, min(xs[i] + ws[i], xs[j] + ws[j]) - max(xs[i], xs[j]) + 1f0)
                ih = max(0f0, min(ys[i] + ws[i], ys[j] + ws[j]) - max(ys[i], ys[j]) + 1f0)
                value = (((iw * ih) / area) < OTHRESHOLD) && (ws[j] != 0f0)
            end
            if !value
                keep[i] = false
                break
            end
        end
    end
    return keep
end

function write_output(path::String, xs, ys, ws, scores, keep, ndetections::Int)
    total = 0
    open(path, "w") do io
        for i in 1:ndetections
            if keep[i] != 0
                @printf(io, "%d,%d,%d,%f\n", round(Int, xs[i]), round(Int, ys[i]), round(Int, ws[i]), scores[i])
                total += 1
            end
        end
    end
    return total
end

function main()
    if length(ARGS) < 3
        println()
        println("Usage: nmstest  <detections.txt>  <output.txt> <repeat>")
        return
    end
    input, output, repeat_s = ARGS[1], ARGS[2], ARGS[3]
    check = "--check" in ARGS[4:end]
    repeat = parse(Int, repeat_s)

    xs, ys, ws, scores, ndetections = read_detections(input)
    @printf("Number of detections read from input file (%s): %d\n", input, ndetections)

    d_xs = CuArray(xs)
    d_ys = CuArray(ys)
    d_ws = CuArray(ws)
    d_scores = CuArray(scores)
    d_bitmap = CUDA.fill(UInt8(1), Int(MAX_DETECTIONS) * Int(MAX_DETECTIONS))
    d_pointsbitmap = CUDA.zeros(UInt8, Int(MAX_DETECTIONS))

    limit = get_upper_limit(Int32(ndetections), Int32(16))
    tx = get_optimal_dim(limit)
    ty = get_optimal_dim(limit)
    grid = (Int(limit ÷ tx), Int(limit ÷ ty))

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=(Int(tx), Int(ty)) blocks=grid generate_nms_bitmap_kernel!(
            d_xs, d_ys, d_ws, d_scores, d_bitmap, limit)
    end
    CUDA.synchronize()
    t1 = time_ns()
    @printf("Average kernel execution time (generate_nms_bitmap): %f (s)\n",
            ((t1 - t0) * 1e-9) / repeat)

    CUDA.synchronize()
    t2 = time_ns()
    for _ in 1:repeat
        @cuda threads=128 blocks=ndetections reduce_nms_bitmap_kernel!(
            d_bitmap, d_pointsbitmap, Int32(ndetections))
    end
    CUDA.synchronize()
    t3 = time_ns()
    @printf("Average kernel execution time (reduce_nms_bitmap): %f (s)\n",
            ((t3 - t2) * 1e-9) / repeat)

    pointsbitmap = Array(d_pointsbitmap)
    total = write_output(output, xs, ys, ws, scores, pointsbitmap, ndetections)
    @printf("Detections after NMS: %d\n", total)

    if check
        expected = cpu_nms(xs, ys, ws, scores, ndetections)
        ok = all((pointsbitmap[i] != 0) == expected[i] for i in 1:ndetections)
        println(ok ? "PASS" : "FAIL")
        ok || exit(1)
    end
end

main()
