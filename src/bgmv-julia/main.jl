using CUDA
using Printf
using Random

Base.@kwdef mutable struct Config
    op::String = "bgmv_shrink"
    num_tokens::Int = 128
    hidden_size::Int = 4096
    lora_rank::Int = 16
    num_loras::Int = 4
    repeat::Int = 200
    add_to_output::Bool = false
    scaling::Float32 = 1.0f0
    vectorize::Bool = false
end

function parse_args(args)
    cfg = Config()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--op" && i < length(args)
            i += 1; cfg.op = args[i]
        elseif arg == "--tokens" && i < length(args)
            i += 1; cfg.num_tokens = parse(Int, args[i])
        elseif arg == "--hidden" && i < length(args)
            i += 1; cfg.hidden_size = parse(Int, args[i])
        elseif arg == "--rank" && i < length(args)
            i += 1; cfg.lora_rank = parse(Int, args[i])
        elseif arg == "--loras" && i < length(args)
            i += 1; cfg.num_loras = parse(Int, args[i])
        elseif arg == "--repeat" && i < length(args)
            i += 1; cfg.repeat = parse(Int, args[i])
        elseif arg == "--scaling" && i < length(args)
            i += 1; cfg.scaling = parse(Float32, args[i])
        elseif arg == "--add_to_output"
            cfg.add_to_output = true
        elseif arg == "--vectorize"
            cfg.vectorize = true
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    return cfg
end

function bgmv_shrink_kernel!(output, input, weights, lora_indices,
                             num_tokens::Int32, hidden_size::Int32,
                             lora_rank::Int32, scaling::Float32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = num_tokens * lora_rank
    if idx0 < total
        token = idx0 ÷ lora_rank
        rank = idx0 % lora_rank
        lora = @inbounds lora_indices[token + Int32(1)] - Int32(1)
        acc = 0.0f0
        for k in Int32(0):(hidden_size - Int32(1))
            inp = Float32(@inbounds input[token * hidden_size + k + Int32(1)])
            wt = Float32(@inbounds weights[lora * lora_rank * hidden_size + rank * hidden_size + k + Int32(1)])
            acc += inp * wt
        end
        @inbounds output[idx0 + Int32(1)] += scaling * acc
    end
    return
end

function bgmv_expand_kernel!(output, input, weights, lora_indices,
                             num_tokens::Int32, hidden_size::Int32,
                             lora_rank::Int32, add_to_output::Bool)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = num_tokens * hidden_size
    if idx0 < total
        token = idx0 ÷ hidden_size
        hidden = idx0 % hidden_size
        lora = @inbounds lora_indices[token + Int32(1)] - Int32(1)
        acc = 0.0f0
        for r in Int32(0):(lora_rank - Int32(1))
            inp = @inbounds input[token * lora_rank + r + Int32(1)]
            wt = Float32(@inbounds weights[lora * hidden_size * lora_rank + hidden * lora_rank + r + Int32(1)])
            acc += inp * wt
        end
        prev = add_to_output ? Float32(@inbounds output[idx0 + Int32(1)]) : 0.0f0
        @inbounds output[idx0 + Int32(1)] = Float16(prev + acc)
    end
    return
end

function ref_bgmv_shrink(input, weights, lora_indices, cfg::Config)
    output = zeros(Float32, cfg.num_tokens * cfg.lora_rank)
    for t in 0:(cfg.num_tokens - 1)
        lid = lora_indices[t + 1] - 1
        for r in 0:(cfg.lora_rank - 1)
            acc = 0.0f0
            for k in 0:(cfg.hidden_size - 1)
                acc += input[t * cfg.hidden_size + k + 1] *
                       weights[lid * cfg.lora_rank * cfg.hidden_size + r * cfg.hidden_size + k + 1]
            end
            output[t * cfg.lora_rank + r + 1] += cfg.scaling * acc
        end
    end
    return output
end

function ref_bgmv_expand(input, weights, lora_indices, cfg::Config)
    output = zeros(Float32, cfg.num_tokens * cfg.hidden_size)
    for t in 0:(cfg.num_tokens - 1)
        lid = lora_indices[t + 1] - 1
        for h in 0:(cfg.hidden_size - 1)
            acc = 0.0f0
            for r in 0:(cfg.lora_rank - 1)
                acc += input[t * cfg.lora_rank + r + 1] *
                       weights[lid * cfg.hidden_size * cfg.lora_rank + h * cfg.lora_rank + r + 1]
            end
            output[t * cfg.hidden_size + h + 1] = acc
        end
    end
    return output
end

function print_header(cfg::Config)
    println()
    println("=== BGMV Benchmark ===")
    println("  op          : $(cfg.op)")
    println("  num_tokens  : $(cfg.num_tokens)")
    println("  hidden_size : $(cfg.hidden_size)")
    println("  lora_rank   : $(cfg.lora_rank)")
    println("  num_loras   : $(cfg.num_loras)")
    println("  repeat      : $(cfg.repeat)")
    println("  add_to_output  : $(cfg.add_to_output ? "true" : "false")")
    println("------------------------------------------------------------------")
end

function run_shrink(cfg::Config, rng)
    input_f32 = rand(rng, Float32, cfg.num_tokens * cfg.hidden_size) .* 2 .- 1
    weights_f32 = rand(rng, Float32, cfg.num_loras * cfg.lora_rank * cfg.hidden_size) .* 2 .- 1
    lora_indices = rand(rng, 1:cfg.num_loras, cfg.num_tokens)
    reference = ref_bgmv_shrink(input_f32, weights_f32, lora_indices, cfg)

    d_input = CuArray(Float16.(input_f32))
    d_weights = CuArray(Float16.(weights_f32))
    d_indices = CuArray(Int32.(lora_indices))
    d_output = CUDA.zeros(Float32, cfg.num_tokens * cfg.lora_rank)
    threads = 256
    blocks = cld(length(d_output), threads)

    @cuda threads=threads blocks=blocks bgmv_shrink_kernel!(
        d_output, d_input, d_weights, d_indices, Int32(cfg.num_tokens),
        Int32(cfg.hidden_size), Int32(cfg.lora_rank), cfg.scaling)
    CUDA.synchronize()
    gpu = Array(d_output)
    max_err = maximum(abs.(gpu .- reference))
    @printf("block_size %4d | correctness check max_err = %e  => %s\n",
            threads, max_err, max_err < 0.1f0 ? "PASS" : "FAIL")

    CUDA.fill!(d_output, 0.0f0)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:cfg.repeat
        @cuda threads=threads blocks=blocks bgmv_shrink_kernel!(
            d_output, d_input, d_weights, d_indices, Int32(cfg.num_tokens),
            Int32(cfg.hidden_size), Int32(cfg.lora_rank), cfg.scaling)
    end
    CUDA.synchronize()
    elapsed_ns = time_ns() - start
    bytes_read = length(input_f32) * sizeof(Float16) + length(weights_f32) * sizeof(Float16) + cfg.num_tokens * sizeof(Int32)
    bytes_write = length(reference) * sizeof(Float32)
    bw = (bytes_read + bytes_write) * cfg.repeat / elapsed_ns
    @printf("bgmv_shrink  | block_size %4d | avg latency: %.3f us  (over %d iters) | bandwidth (approx): %.1f GB/s\n",
            threads, elapsed_ns * 1.0e-3 / cfg.repeat, cfg.repeat, bw)
    return max_err < 0.1f0
end

function run_expand(cfg::Config, rng)
    input_f32 = rand(rng, Float32, cfg.num_tokens * cfg.lora_rank) .* 2 .- 1
    weights_f32 = rand(rng, Float32, cfg.num_loras * cfg.hidden_size * cfg.lora_rank) .* 2 .- 1
    lora_indices = rand(rng, 1:cfg.num_loras, cfg.num_tokens)
    reference = ref_bgmv_expand(input_f32, weights_f32, lora_indices, cfg)

    d_input = CuArray(input_f32)
    d_weights = CuArray(Float16.(weights_f32))
    d_indices = CuArray(Int32.(lora_indices))
    d_output = CUDA.zeros(Float16, cfg.num_tokens * cfg.hidden_size)
    threads = 256
    blocks = cld(length(d_output), threads)

    @cuda threads=threads blocks=blocks bgmv_expand_kernel!(
        d_output, d_input, d_weights, d_indices, Int32(cfg.num_tokens),
        Int32(cfg.hidden_size), Int32(cfg.lora_rank), cfg.add_to_output)
    CUDA.synchronize()
    gpu = Float32.(Array(d_output))
    max_err = maximum(abs.(gpu .- reference))
    @printf("block_size %4d | correctness check max_err = %e  => %s\n",
            threads, max_err, max_err < 0.1f0 ? "PASS" : "FAIL")

    CUDA.fill!(d_output, Float16(0))
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:cfg.repeat
        @cuda threads=threads blocks=blocks bgmv_expand_kernel!(
            d_output, d_input, d_weights, d_indices, Int32(cfg.num_tokens),
            Int32(cfg.hidden_size), Int32(cfg.lora_rank), cfg.add_to_output)
    end
    CUDA.synchronize()
    elapsed_ns = time_ns() - start
    bytes_read = length(input_f32) * sizeof(Float32) + length(weights_f32) * sizeof(Float16) + cfg.num_tokens * sizeof(Int32)
    bytes_write = length(reference) * sizeof(Float16)
    bw = (bytes_read + bytes_write) * cfg.repeat / elapsed_ns
    @printf("bgmv_expand | block_size %4d | add_to_output=%s  avg latency: %.3f us | bandwidth (approx): %.1f GB/s\n",
            threads, cfg.add_to_output ? "true" : "false", elapsed_ns * 1.0e-3 / cfg.repeat, bw)
    return max_err < 0.1f0
end

function main(args)
    cfg = parse_args(args)
    print_header(cfg)
    rng = MersenneTwister(19937)
    ok = cfg.op == "bgmv_shrink" ? run_shrink(cfg, rng) :
         cfg.op == "bgmv_expand" ? run_expand(cfg, rng) :
         error("Unknown op $(cfg.op)")
    println("------------------------------------------------------------------")
    return ok ? 0 : 1
end

exit(main(ARGS))
