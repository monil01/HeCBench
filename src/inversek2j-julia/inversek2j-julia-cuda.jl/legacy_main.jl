# Julia (CUDA.jl) port of `inversek2j` HeCBench benchmark.
#
# Inverse kinematics for a 3-joint planar arm, cyclic coordinate descent.
# Each thread handles one (x, y) target; the small joint state is kept in
# scalar registers. Verifies against a pure-Julia CPU reference.
#
# Usage: julia main.jl <coord_in.txt> <iterations>

using CUDA
using Printf

const MAX_LOOP = 25
const MAX_DIFF = 0.15f0
const NUM_JOINTS = 3
const BLOCK_SIZE = 128


function invkin_kernel!(xTgt::CuDeviceVector{Float32}, yTgt::CuDeviceVector{Float32},
                       angles::CuDeviceVector{Float32}, size::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx > size
        return
    end

    curr_xTargetIn = xTgt[idx]
    curr_yTargetIn = yTgt[idx]

    # Angles a0, a1, a2 (initial: 0.0)
    a0 = 0.0f0; a1 = 0.0f0; a2 = 0.0f0

    # xData[0..3] = i, yData[0..3] = 0. Only end xData/yData (index 3) is
    # updated by the FK step.
    for curr_loop in 1:MAX_LOOP
        # Iter = 3, 2, 1
        for iter in NUM_JOINTS:-1:1
            # Forward-kinematics — recompute joint positions from current angles.
            # Joint 0 is at (0,0). Segment lengths are 1 each.
            x0 = 0.0f0; y0 = 0.0f0
            x1 = x0 + CUDA.cos(a0);              y1 = y0 + CUDA.sin(a0)
            x2 = x1 + CUDA.cos(a0 + a1);         y2 = y1 + CUDA.sin(a0 + a1)
            x3 = x2 + CUDA.cos(a0 + a1 + a2);    y3 = y2 + CUDA.sin(a0 + a1 + a2)

            pe_x = x3; pe_y = y3
            if iter == 1
                pc_x = x0; pc_y = y0
            elseif iter == 2
                pc_x = x1; pc_y = y1
            else
                pc_x = x2; pc_y = y2
            end

            diff_pe_pc_x  = pe_x - pc_x
            diff_pe_pc_y  = pe_y - pc_y
            diff_tgt_pc_x = curr_xTargetIn - pc_x
            diff_tgt_pc_y = curr_yTargetIn - pc_y
            len_pe_pc  = CUDA.sqrt(diff_pe_pc_x^2 + diff_pe_pc_y^2)
            len_tgt_pc = CUDA.sqrt(diff_tgt_pc_x^2 + diff_tgt_pc_y^2)
            if len_pe_pc <= 0.0f0 || len_tgt_pc <= 0.0f0
                continue
            end
            a_x = diff_pe_pc_x / len_pe_pc
            a_y = diff_pe_pc_y / len_pe_pc
            b_x = diff_tgt_pc_x / len_tgt_pc
            b_y = diff_tgt_pc_y / len_tgt_pc

            cos_v = a_x * b_x + a_y * b_y
            cos_v = min(max(cos_v, -1.0f0), 1.0f0)
            sin_v = a_x * b_y - a_y * b_x
            angle = CUDA.acos(cos_v)
            if sin_v < 0.0f0
                angle = -angle
            end
            if iter == 1
                a0 += angle
            elseif iter == 2
                a1 += angle
            else
                a2 += angle
            end
        end
    end

    angles[3*(idx - Int32(1)) + Int32(1)] = a0
    angles[3*(idx - Int32(1)) + Int32(2)] = a1
    angles[3*(idx - Int32(1)) + Int32(3)] = a2
    return
end


function invkin_cpu(xTgt::Vector{Float32}, yTgt::Vector{Float32})::Vector{Float32}
    n = length(xTgt)
    out = Vector{Float32}(undef, 3 * n)
    for idx in 1:n
        curr_x = xTgt[idx]; curr_y = yTgt[idx]
        a0 = 0.0f0; a1 = 0.0f0; a2 = 0.0f0
        for _ in 1:MAX_LOOP
            for iter in 3:-1:1
                x0 = 0.0f0; y0 = 0.0f0
                x1 = x0 + Float32(cos(a0));        y1 = y0 + Float32(sin(a0))
                x2 = x1 + Float32(cos(a0 + a1));   y2 = y1 + Float32(sin(a0 + a1))
                x3 = x2 + Float32(cos(a0+a1+a2));  y3 = y2 + Float32(sin(a0+a1+a2))
                pe_x = x3; pe_y = y3
                pc_x, pc_y = iter == 1 ? (x0, y0) : iter == 2 ? (x1, y1) : (x2, y2)
                dx1 = pe_x - pc_x; dy1 = pe_y - pc_y
                dx2 = curr_x - pc_x; dy2 = curr_y - pc_y
                l1 = Float32(sqrt(dx1*dx1 + dy1*dy1))
                l2 = Float32(sqrt(dx2*dx2 + dy2*dy2))
                (l1 <= 0.0f0 || l2 <= 0.0f0) && continue
                ax = dx1 / l1; ay = dy1 / l1
                bx = dx2 / l2; by = dy2 / l2
                cv = min(max(ax*bx + ay*by, -1.0f0), 1.0f0)
                sv = ax*by - ay*bx
                ang = Float32(acos(cv))
                sv < 0 && (ang = -ang)
                iter == 1 && (a0 += ang)
                iter == 2 && (a1 += ang)
                iter == 3 && (a2 += ang)
            end
        end
        out[3*(idx-1)+1] = a0
        out[3*(idx-1)+2] = a1
        out[3*(idx-1)+3] = a2
    end
    return out
end


function main()
    if length(ARGS) < 2
        println("Usage: main.jl <coord_in.txt> <iterations>")
        return 1
    end
    fn = ARGS[1]
    reps = parse(Int, ARGS[2])

    xs = Float32[]; ys = Float32[]
    open(fn) do io
        n = parse(Int, strip(readline(io)))
        for _ in 1:n
            line = readline(io); parts = split(line)
            push!(xs, parse(Float32, parts[1]))
            push!(ys, parse(Float32, parts[2]))
        end
    end
    n = length(xs)
    @printf("Number of targets: %d\n", n)

    d_x = CuArray(xs)
    d_y = CuArray(ys)
    d_a = CuArray(zeros(Float32, 3 * n))

    blocks = cld(n, BLOCK_SIZE)

    # Time
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:reps
        @cuda threads=BLOCK_SIZE blocks=blocks invkin_kernel!(d_x, d_y, d_a, Int32(n))
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) / 1e6 / reps
    @printf("Average kernel execution time %f (ms)\n", elapsed_ms)

    gpu = Array(d_a)
    # CPU reference on a subset (first 4096 rows) — full compute is O(n*MAX_LOOP*3)
    subset = min(4096, n)
    ref = invkin_cpu(xs[1:subset], ys[1:subset])
    max_err = 0.0f0
    for i in 1:(3*subset)
        e = abs(gpu[i] - ref[i])
        max_err = max(max_err, e)
    end
    @printf("Max angle error on first %d targets: %.6f\n", subset, max_err)
    # MAX_DIFF from the CUDA benchmark (`main.cu`) — CCD-derived angles can
    # differ by ~0.02 rad and still describe the same end-effector position.
    println(max_err <= 0.15 ? "PASS" : "FAIL")
    return 0
end

main()
