using CUDA
using Printf
using Random

const FP = Float32
const NUMBER_PAR_PER_BOX = 100
const NUMBER_THREADS = 128

function is_integer_string(s)
    return occursin(r"^-?[0-9]+$", s)
end

function parse_args(args)
    boxes1d = 1
    if length(args) == 2 && args[1] == "-boxes1d" && is_integer_string(args[2])
        boxes1d = parse(Int, args[2])
        boxes1d < 0 && (println("ERROR: Wrong value to -boxes1d argument, cannot be <=0"); exit(0))
    elseif length(args) == 2 && args[1] == "-boxes1d"
        println("ERROR: Value to -boxes1d argument in not a number")
        exit(0)
    elseif isempty(args)
        println("Provide boxes1d argument, example: -boxes1d 16")
        exit(0)
    else
        println("ERROR: Unknown argument")
        exit(0)
    end
    return boxes1d
end

function build_boxes(boxes1d::Int)
    number_boxes = boxes1d^3
    offsets = Vector{Int32}(undef, number_boxes)
    nn = zeros(Int32, number_boxes)
    nei_numbers = fill(Int32(0), 26, number_boxes)
    nh = 1
    @inbounds for z in 0:boxes1d-1, y in 0:boxes1d-1, x in 0:boxes1d-1
        offsets[nh] = Int32((nh - 1) * NUMBER_PAR_PER_BOX)
        count = 0
        for dz in -1:1, dy in -1:1, dx in -1:1
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < boxes1d && 0 <= ny < boxes1d && 0 <= nz < boxes1d && !(dx == 0 && dy == 0 && dz == 0)
                count += 1
                nei_numbers[count, nh] = Int32(nz * boxes1d * boxes1d + ny * boxes1d + nx)
            end
        end
        nn[nh] = Int32(count)
        nh += 1
    end
    return offsets, nn, nei_numbers
end

function init_inputs(space_elem::Int)
    rng = MersenneTwister(2)
    rv_v = Vector{FP}(undef, space_elem)
    rv_x = Vector{FP}(undef, space_elem)
    rv_y = Vector{FP}(undef, space_elem)
    rv_z = Vector{FP}(undef, space_elem)
    qv = Vector{FP}(undef, space_elem)
    @inbounds for i in 1:space_elem
        rv_v[i] = FP(rand(rng, 1:10) / 10)
        rv_x[i] = FP(rand(rng, 1:10) / 10)
        rv_y[i] = FP(rand(rng, 1:10) / 10)
        rv_z[i] = FP(rand(rng, 1:10) / 10)
    end
    @inbounds for i in 1:space_elem
        qv[i] = FP(rand(rng, 1:10) / 10)
    end
    return rv_v, rv_x, rv_y, rv_z, qv
end

function md_cpu(offsets, nn, nei_numbers, rv_v, rv_x, rv_y, rv_z, qv, alpha::FP)
    fv_v = zeros(FP, length(qv))
    fv_x = zeros(FP, length(qv))
    fv_y = zeros(FP, length(qv))
    fv_z = zeros(FP, length(qv))
    a2 = FP(2) * alpha * alpha
    @inbounds for bx in eachindex(offsets)
        first_i = Int(offsets[bx]) + 1
        for kk in 0:Int(nn[bx])
            pointer = kk == 0 ? bx : Int(nei_numbers[kk, bx]) + 1
            first_j = Int(offsets[pointer]) + 1
            for i in 0:NUMBER_PAR_PER_BOX-1
                ii = first_i + i
                for j in 0:NUMBER_PAR_PER_BOX-1
                    jj = first_j + j
                    r2 = rv_v[ii] + rv_v[jj] - (rv_x[ii] * rv_x[jj] + rv_y[ii] * rv_y[jj] + rv_z[ii] * rv_z[jj])
                    u2 = a2 * r2
                    vij = exp(-u2)
                    fs = FP(2) * vij
                    dx = rv_x[ii] - rv_x[jj]
                    dy = rv_y[ii] - rv_y[jj]
                    dz = rv_z[ii] - rv_z[jj]
                    fv_v[ii] += qv[jj] * vij
                    fv_x[ii] += qv[jj] * fs * dx
                    fv_y[ii] += qv[jj] * fs * dy
                    fv_z[ii] += qv[jj] * fs * dz
                end
            end
        end
    end
    return fv_v, fv_x, fv_y, fv_z
end

function md_cpu_for_boxes(offsets, nn, nei_numbers, rv_v, rv_x, rv_y, rv_z, qv,
                          alpha::FP, boxes)
    fv_v = zeros(FP, length(qv))
    fv_x = zeros(FP, length(qv))
    fv_y = zeros(FP, length(qv))
    fv_z = zeros(FP, length(qv))
    a2 = FP(2) * alpha * alpha
    @inbounds for bx in boxes
        first_i = Int(offsets[bx]) + 1
        for kk in 0:Int(nn[bx])
            pointer = kk == 0 ? bx : Int(nei_numbers[kk, bx]) + 1
            first_j = Int(offsets[pointer]) + 1
            for i in 0:NUMBER_PAR_PER_BOX-1
                ii = first_i + i
                for j in 0:NUMBER_PAR_PER_BOX-1
                    jj = first_j + j
                    r2 = rv_v[ii] + rv_v[jj] - (rv_x[ii] * rv_x[jj] + rv_y[ii] * rv_y[jj] + rv_z[ii] * rv_z[jj])
                    vij = exp(-(a2 * r2))
                    fs = FP(2) * vij
                    dx = rv_x[ii] - rv_x[jj]
                    dy = rv_y[ii] - rv_y[jj]
                    dz = rv_z[ii] - rv_z[jj]
                    fv_v[ii] += qv[jj] * vij
                    fv_x[ii] += qv[jj] * fs * dx
                    fv_y[ii] += qv[jj] * fs * dy
                    fv_z[ii] += qv[jj] * fs * dz
                end
            end
        end
    end
    return fv_v, fv_x, fv_y, fv_z
end

function md_kernel!(offsets, nn, nei_numbers, rv_v, rv_x, rv_y, rv_z, qv,
                    fv_v, fv_x, fv_y, fv_z, alpha::FP, number_boxes::Int32)
    bx0 = blockIdx().x - Int32(1)
    tx0 = threadIdx().x - Int32(1)
    if bx0 < number_boxes
        first_i0 = @inbounds offsets[bx0 + Int32(1)]
        a2 = FP(2) * alpha * alpha
        kk = Int32(0)
        while kk <= @inbounds(nn[bx0 + Int32(1)])
            pointer0 = kk == Int32(0) ? bx0 : @inbounds(nei_numbers[kk + Int32(0), bx0 + Int32(1)])
            first_j0 = @inbounds offsets[pointer0 + Int32(1)]
            wtx = tx0
            while wtx < Int32(NUMBER_PAR_PER_BOX)
                ii = first_i0 + wtx + Int32(1)
                j = Int32(0)
                while j < Int32(NUMBER_PAR_PER_BOX)
                    jj = first_j0 + j + Int32(1)
                    r2 = @inbounds(rv_v[ii]) + @inbounds(rv_v[jj]) -
                         (@inbounds(rv_x[ii]) * @inbounds(rv_x[jj]) +
                          @inbounds(rv_y[ii]) * @inbounds(rv_y[jj]) +
                          @inbounds(rv_z[ii]) * @inbounds(rv_z[jj]))
                    vij = exp(-(a2 * r2))
                    fs = FP(2) * vij
                    dx = @inbounds(rv_x[ii]) - @inbounds(rv_x[jj])
                    dy = @inbounds(rv_y[ii]) - @inbounds(rv_y[jj])
                    dz = @inbounds(rv_z[ii]) - @inbounds(rv_z[jj])
                    q = @inbounds qv[jj]
                    @inbounds fv_v[ii] += q * vij
                    @inbounds fv_x[ii] += q * fs * dx
                    @inbounds fv_y[ii] += q * fs * dy
                    @inbounds fv_z[ii] += q * fs * dz
                    j += Int32(1)
                end
                wtx += Int32(NUMBER_THREADS)
            end
            kk += Int32(1)
        end
    end
    return
end

function max_rel_error(refs, vals; indices=nothing)
    maxerr = 0.0
    for (ref, val) in zip(refs, vals)
        iter = indices === nothing ? eachindex(ref) : indices
        @inbounds for i in iter
            denom = max(1.0, abs(Float64(ref[i])))
            maxerr = max(maxerr, abs(Float64(ref[i] - val[i])) / denom)
        end
    end
    return maxerr
end

function main()
    println("WG size of kernel = ", NUMBER_THREADS, " ")
    boxes1d = parse_args(ARGS)
    println("Configuration used: arch = 0, cores = 1, boxes1d = ", boxes1d)
    alpha = FP(0.5)
    number_boxes = boxes1d^3
    space_elem = number_boxes * NUMBER_PAR_PER_BOX
    offsets, nn, nei_numbers = build_boxes(boxes1d)
    rv_v, rv_x, rv_y, rv_z, qv = init_inputs(space_elem)

    start_total = time_ns()
    d_offsets = CuArray(offsets)
    d_nn = CuArray(nn)
    d_nei_numbers = CuArray(nei_numbers)
    d_rv_v = CuArray(rv_v)
    d_rv_x = CuArray(rv_x)
    d_rv_y = CuArray(rv_y)
    d_rv_z = CuArray(rv_z)
    d_qv = CuArray(qv)
    d_fv_v = CUDA.zeros(FP, space_elem)
    d_fv_x = CUDA.zeros(FP, space_elem)
    d_fv_y = CUDA.zeros(FP, space_elem)
    d_fv_z = CUDA.zeros(FP, space_elem)

    CUDA.synchronize()
    kstart = time_ns()
    @cuda threads=NUMBER_THREADS blocks=number_boxes md_kernel!(
        d_offsets, d_nn, d_nei_numbers, d_rv_v, d_rv_x, d_rv_y, d_rv_z, d_qv,
        d_fv_v, d_fv_x, d_fv_y, d_fv_z, alpha, Int32(number_boxes))
    CUDA.synchronize()
    kend = time_ns()

    fv = (Array(d_fv_v), Array(d_fv_x), Array(d_fv_y), Array(d_fv_z))
    total_time = (time_ns() - start_total) * 1.0e-9
    kernel_time = (kend - kstart) * 1.0e-9
    println("Device offloading time:")
    @printf("%.12f s\n", total_time)
    println("Kernel execution time:")
    @printf("%.12f s\n", kernel_time)

    if number_boxes <= 64
        refs = md_cpu(offsets, nn, nei_numbers, rv_v, rv_x, rv_y, rv_z, qv, alpha)
        err = max_rel_error(refs, fv)
    else
        sample_boxes = unique([1, max(1, number_boxes ÷ 2), number_boxes])
        refs = md_cpu_for_boxes(offsets, nn, nei_numbers, rv_v, rv_x, rv_y, rv_z, qv, alpha, sample_boxes)
        sample_indices = Int[]
        for bx in sample_boxes
            append!(sample_indices, (Int(offsets[bx]) + 1):(Int(offsets[bx]) + NUMBER_PAR_PER_BOX))
        end
        err = max_rel_error(refs, fv; indices=sample_indices)
    end
    println(err <= 1.0e-5 ? "PASS" : "FAIL")
    err <= 1.0e-5 || exit(1)
end

main()
