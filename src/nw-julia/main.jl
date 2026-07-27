using CUDA
using Printf

# Julia port of nw-cuda (Needleman-Wunsch).
# Simplification: instead of the CUDA 16x16-tile skewed wavefront, we use a
# straightforward cell-level anti-diagonal wavefront -- one kernel launch per
# anti-diagonal, one thread per cell on that diagonal. Same math; matches CPU
# reference exactly.

const BLOSUM62 = [
     4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4;
    -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4;
    -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4;
    -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4;
     0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4;
    -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4;
    -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4;
     0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4;
    -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4;
    -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4;
    -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4;
    -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4;
    -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4;
    -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4;
    -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4;
     1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4;
     0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4;
    -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4;
    -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4;
     0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4;
    -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4;
    -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4;
     0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4;
    -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
]

# Cell-level diagonal wavefront kernel.
# Processes cells (i,j) with i+j = d, i in [i_min, i_max].
function nw_diag_kernel!(itemsets, ref, max_cols::Int32, penalty::Int32,
                          d::Int32, i_min::Int32, i_max::Int32)
    tid = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    i = i_min + tid - Int32(1)
    if i <= i_max
        j = d - i
        # cell index (i,j) at row-major: i * max_cols + j (0-based)
        c = i * max_cols + j
        cnw = (i - Int32(1)) * max_cols + (j - Int32(1))
        cn  = (i - Int32(1)) * max_cols + j
        cw  = i * max_cols + (j - Int32(1))
        @inbounds begin
            a = itemsets[cnw + Int32(1)] + ref[c + Int32(1)]
            b = itemsets[cw  + Int32(1)] - penalty
            cc = itemsets[cn  + Int32(1)] - penalty
            m = a
            if b > m; m = b; end
            if cc > m; m = cc; end
            itemsets[c + Int32(1)] = m
        end
    end
    return
end

function nw_host!(itemsets, ref, max_cols, penalty)
    for i in 1:(max_cols-1)
        for j in 1:(max_cols-1)
            c = i * max_cols + j
            cnw = (i - 1) * max_cols + (j - 1)
            cn  = (i - 1) * max_cols + j
            cw  = i * max_cols + (j - 1)
            a = itemsets[cnw + 1] + ref[c + 1]
            b = itemsets[cw + 1]  - penalty
            cc = itemsets[cn + 1] - penalty
            m = max(a, b, cc)
            itemsets[c + 1] = m
        end
    end
end

# Simple LCG standing in for srand(7) style seed sequence -- since both CPU and
# GPU use the SAME buffer, verification only cares about determinism.
mutable struct LCG
    s::UInt64
end
next_int!(l::LCG) = (l.s = l.s * UInt64(6364136223846793005) + UInt64(1442695040888963407); Int((l.s >> 33) & UInt64(0x7fffffff)))

function main()
    if length(ARGS) < 3
        println("Usage: main.jl <dim> <penalty> <repeat>")
        return 1
    end
    dim = parse(Int, ARGS[1])
    penalty = parse(Int, ARGS[2])
    repeat_n = parse(Int, ARGS[3])
    if dim % 16 != 0
        println("The dimension values must be a multiple of 16")
        return 1
    end
    @printf("WG size of kernel = 16 \n")

    max_rows = dim + 1
    max_cols = dim + 1
    N = max_rows * max_cols

    input_itemsets = zeros(Int32, N)
    reference_arr  = zeros(Int32, N)

    lcg = LCG(7)
    # boundary vals (must be 1..10 to index blosum62)
    boundary_col = Vector{Int32}(undef, max_rows - 1)
    boundary_row = Vector{Int32}(undef, max_cols - 1)
    for i in 1:(max_rows - 1)
        boundary_col[i] = Int32(next_int!(lcg) % 10 + 1)
    end
    for j in 1:(max_cols - 1)
        boundary_row[j] = Int32(next_int!(lcg) % 10 + 1)
    end
    for i in 1:(max_rows - 1)
        input_itemsets[i * max_cols + 1] = boundary_col[i]  # column 0
    end
    for j in 1:(max_cols - 1)
        input_itemsets[j + 1] = boundary_row[j]  # row 0
    end
    # Build reference using blosum62[col_val+1][row_val+1] (values 1..10)
    for i in 1:(max_rows - 1), j in 1:(max_cols - 1)
        rv = input_itemsets[i * max_cols + 1]  # column boundary at row i
        cv = input_itemsets[j + 1]              # row boundary at col j
        # BLOSUM62 is 24x24 (Julia 1-based indexing). Values in [1,10].
        reference_arr[i * max_cols + j + 1] = Int32(BLOSUM62[rv + 1, cv + 1])
    end
    # Overwrite boundaries with -i*penalty and -j*penalty
    for i in 1:(max_rows - 1)
        input_itemsets[i * max_cols + 1] = Int32(-i * penalty)
    end
    for j in 1:(max_cols - 1)
        input_itemsets[j + 1] = Int32(-j * penalty)
    end

    d_ref = CuArray(reference_arr)
    d_in  = CuArray(copy(input_itemsets))
    initial_in = copy(input_itemsets)

    threads = 256
    CUDA.synchronize()
    t0 = time_ns()

    for _ in 1:repeat_n
        CUDA.copyto!(d_in, initial_in)
        # anti-diagonals d = 2..(2*(max_cols-1)); actual diagonal index = i+j = d
        # We want to process interior cells: i in 1..(max_rows-1), j in 1..(max_cols-1)
        # so d in 2..(2*(max_cols-1))
        for d in 2:(2 * (max_cols - 1))
            i_min = max(1, d - (max_cols - 1))
            i_max = min(max_rows - 1, d - 1)
            n = i_max - i_min + 1
            blocks = cld(n, threads)
            @cuda threads=threads blocks=blocks nw_diag_kernel!(d_in, d_ref,
                Int32(max_cols), Int32(penalty), Int32(d), Int32(i_min), Int32(i_max))
        end
    end
    CUDA.synchronize()
    elapsed = (time_ns() - t0) * 1e-9 / repeat_n
    @printf("Total kernel execution time: %f (s)\n", elapsed)

    output = Array(d_in)

    # CPU reference
    ref_input = copy(initial_in)
    nw_host!(ref_input, reference_arr, max_cols, penalty)
    ok = ref_input == output
    println(ok ? "PASS" : "FAIL")
    return 0
end

main()
