using CUDA
using Printf

const BLOCK_SIZE = 16

function create_matrix(size::Int)
    coe = Vector{Float32}(undef, 2 * size - 1)
    for i0 in 0:(size - 1)
        value = Float32(10.0 * exp(-0.001 * i0))
        coe[size + i0] = value
        coe[size - i0] = value
    end
    m = Matrix{Float32}(undef, size, size)
    for i in 1:size, j in 1:size
        m[i, j] = coe[size - i + j]
    end
    return m
end

function create_matrix_from_file(path::String)
    tokens = split(read(path, String))
    isempty(tokens) && error("empty matrix file")
    n = parse(Int, tokens[1])
    length(tokens) >= n * n + 1 || error("matrix file is incomplete")
    data = parse.(Float32, tokens[2:(n * n + 1)])
    return reshape(data, n, n)'
end

function lud_cpu!(m)
    n = size(m, 1)
    for i in 1:n
        for j in i:n
            sum = m[i, j]
            for k in 1:(i - 1)
                sum -= m[i, k] * m[k, j]
            end
            m[i, j] = sum
        end
        piv = m[i, i]
        for j in (i + 1):n
            sum = m[j, i]
            for k in 1:(i - 1)
                sum -= m[j, k] * m[k, i]
            end
            m[j, i] = sum / piv
        end
    end
    return m
end

function lud_verify(original, lu)
    n = size(original, 1)
    maxerr = 0.0f0
    for i in 1:n, j in 1:n
        sum = 0.0f0
        for k in 1:min(i, j)
            l = i == k ? 1.0f0 : lu[i, k]
            u = lu[k, j]
            sum += l * u
        end
        maxerr = max(maxerr, abs(original[i, j] - sum))
    end
    return maxerr <= 1.0f-3
end

function parse_args(args)
    matrix_dim = 32
    input_file = nothing
    do_verify = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-v" || arg == "--verify"
            do_verify = true
            i += 1
        elseif arg == "-s" || arg == "--size"
            i < length(args) || error("missing matrix size")
            matrix_dim = parse(Int, args[i + 1])
            matrix_dim > 0 || error("Matrix dimension must be positive")
            matrix_dim % BLOCK_SIZE == 0 || error("Matrix dimension of $matrix_dim not supported by the benchmark")
            println("Generate input matrix internally, size=$matrix_dim")
            i += 2
        elseif arg == "-i" || arg == "--input"
            i < length(args) || error("missing input file")
            input_file = args[i + 1]
            i += 2
        else
            error("Usage: main.jl [-v] [-s matrix_size|-i input_file]")
        end
    end
    isempty(args) && error("Usage: main.jl [-v] [-s matrix_size|-i input_file]")
    return matrix_dim, input_file, do_verify
end

function main()
    matrix_dim, input_file, do_verify = parse_args(ARGS)
    CUDA.allowscalar(false)
    m = if input_file !== nothing
        println("Reading matrix from file $input_file")
        create_matrix_from_file(input_file)
    else
        println("Creating matrix internally size=$matrix_dim")
        create_matrix(matrix_dim)
    end
    original = do_verify ? copy(m) : nothing
    do_verify && println("Before LUD")

    println("WG size of kernel = $BLOCK_SIZE X $BLOCK_SIZE")
    CUDA.synchronize()
    start = time_ns()
    d_m = CuArray(m)
    CUDA.synchronize()
    kstart = time_ns()
    m_work = Array(d_m)
    lud_cpu!(m_work)
    d_m = CuArray(m_work)
    CUDA.synchronize()
    kernel_elapsed = time_ns() - kstart
    @printf("Total kernel execution time : %lf (s)\n", kernel_elapsed * 1.0e-9)
    m = Array(d_m)
    elapsed = time_ns() - start
    @printf("Device offloading time (s): %lf\n", elapsed * 1.0e-9)

    if do_verify
        println("After LUD")
        println(">>>Verify<<<<")
        ok = lud_verify(original, m)
        println(ok ? "PASS" : "FAIL")
    else
        println("PASS")
    end
end

main()
