using CUDA
using Printf
using Random

const THREADS_X = 32

function read_mtx(path::String)
    rows = Int[]
    cols = Int[]
    vals = Float32[]
    nrows = 0
    ncols = 0
    nnz = 0
    open(path, "r") do io
        got_header = false
        for raw in eachline(io)
            line = strip(raw)
            if isempty(line) || startswith(line, "%")
                continue
            end
            parts = split(line)
            if !got_header
                nrows = parse(Int, parts[1])
                ncols = parse(Int, parts[2])
                nnz = parse(Int, parts[3])
                got_header = true
            else
                push!(rows, parse(Int, parts[1]) - 1)
                push!(cols, parse(Int, parts[2]) - 1)
                push!(vals, length(parts) >= 3 ? parse(Float32, parts[3]) : 1.0f0)
            end
        end
    end
    return rows, cols, vals, nrows, ncols, nnz
end

function coo_to_csr(row_indices, col_indices, nrows::Int, nnz::Int)
    rowptr = zeros(Int32, nrows + 1)
    cols = Vector{Int32}(undef, nnz)
    vals = ones(Float32, nnz)
    @inbounds for r in row_indices
        rowptr[r + 2] += Int32(1)
    end
    @inbounds for i in 2:nrows+1
        rowptr[i] += rowptr[i - 1]
    end
    next = copy(rowptr)
    @inbounds for n in 1:nnz
        r = row_indices[n] + 1
        pos = Int(next[r]) + 1
        cols[pos] = Int32(col_indices[n])
        next[r] += Int32(1)
    end
    return rowptr, cols, vals
end

@inline function lanes_for_method(method::Int32)
    return method == 1 ? Int32(32) :
           method == 2 ? Int32(64) :
           method == 3 ? Int32(128) : Int32(256)
end

function spmm_kernel!(nrows::Int32, bncols::Int32, rowptr, colind, aval, b, c, method::Int32)
    lane = Int32(threadIdx().x - 1)
    row = Int32((blockIdx().x - 1) * blockDim().y + threadIdx().y - 1)
    group = Int32(blockIdx().y - 1)
    lanes = lanes_for_method(method)
    base_col = group * lanes + lane
    stride = Int32(32)
    while base_col < min((group + Int32(1)) * lanes, bncols)
        if row < nrows
            acc = 0.0f0
            lo = @inbounds rowptr[row + 1]
            hi = @inbounds rowptr[row + 2]
            for ptr in lo:hi-Int32(1)
                col = @inbounds colind[ptr + Int32(1)]
                acc += (@inbounds aval[ptr + Int32(1)]) * (@inbounds b[col * bncols + base_col + Int32(1)])
            end
            @inbounds c[row * bncols + base_col + Int32(1)] = acc
        end
        base_col += stride
    end
    return
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <matrix file> <tile row> <repeat>")
        return 1
    end
    matrix_file = args[1]
    tile_row = parse(Int, args[2])
    repeat = parse(Int, args[3])
    max_ncols = 256

    println("reading data file ...")
    row_indices, col_indices, _values, nrows, ncols, nnz = read_mtx(matrix_file)
    rowptr, colind, aval = coo_to_csr(row_indices, col_indices, nrows, nnz)
    @printf("read file ok. N=%d nnz=%d\n", nrows, nnz)

    Random.seed!(123)
    b = Float32.((rand(0:99, max_ncols * ncols) .- 50) ./ 100)
    golden = zeros(Float32, nrows * max_ncols)
    @inbounds for i in 0:nrows-1, k in 0:max_ncols-1
        acc = 0.0f0
        for ptr in Int(rowptr[i + 1]):Int(rowptr[i + 2])-1
            acc += aval[ptr + 1] * b[max_ncols * Int(colind[ptr + 1]) + k + 1]
        end
        golden[max_ncols * i + k + 1] = acc
    end

    d_rowptr = CuArray(rowptr)
    d_colind = CuArray(colind)
    d_aval = CuArray(aval)
    ok = true
    for bncols in (256,)
        d_b = CuArray(b)
        for method in 1:4
            d_c = CUDA.zeros(Float32, nrows * bncols)
            lanes = Int(lanes_for_method(Int32(method)))
            blocks = (max(cld(nrows, tile_row), 1), max(cld(bncols, lanes), 1))
            CUDA.synchronize()
            t0 = time_ns()
            for _ in 1:repeat
                @cuda threads=(THREADS_X, tile_row) blocks=blocks spmm_kernel!(
                    Int32(nrows), Int32(bncols), d_rowptr, d_colind, d_aval, d_b, d_c, Int32(method))
            end
            CUDA.synchronize()
            @printf("Average kernel (method %d) execution time %f (us)\n",
                    method, (time_ns() - t0) * 1.0e-3 / repeat)
            c = Array(d_c)
            for i in 0:nrows-1, j in 0:bncols-1
                if abs(c[i * bncols + j + 1] - golden[i * max_ncols + j + 1]) > 1.0f-2
                    @printf("b_ncols %d kernel method %d: results mismatch %f %f\n",
                            bncols, method, c[i * bncols + j + 1], golden[i * max_ncols + j + 1])
                    ok = false
                    break
                end
            end
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
