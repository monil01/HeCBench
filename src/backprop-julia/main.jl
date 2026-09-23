using CUDA
using Printf
using Random

const ETA = 0.3f0
const MOMENTUM = 0.3f0
const HID = 16
const OUT = 1

sigmoid(x) = 1f0 / (1f0 + exp(-x))

mutable struct BPNN
    input_n::Int
    hidden_n::Int
    output_n::Int
    input_units::Vector{Float32}
    hidden_units::Vector{Float32}
    output_units::Vector{Float32}
    hidden_delta::Vector{Float32}
    output_delta::Vector{Float32}
    target::Vector{Float32}
    input_weights::Matrix{Float32}
    hidden_weights::Matrix{Float32}
    input_prev_weights::Matrix{Float32}
    hidden_prev_weights::Matrix{Float32}
end

function create_net(n_in::Int, n_hidden::Int, n_out::Int, rng::AbstractRNG)
    input_units = zeros(Float32, n_in + 1)
    hidden_units = zeros(Float32, n_hidden + 1)
    output_units = zeros(Float32, n_out + 1)
    hidden_delta = zeros(Float32, n_hidden + 1)
    output_delta = zeros(Float32, n_out + 1)
    target = fill(0.1f0, n_out + 1)
    input_weights = rand(rng, Float32, n_in + 1, n_hidden + 1)
    hidden_weights = rand(rng, Float32, n_hidden + 1, n_out + 1)
    input_prev_weights = zeros(Float32, n_in + 1, n_hidden + 1)
    hidden_prev_weights = zeros(Float32, n_hidden + 1, n_out + 1)
    return BPNN(n_in, n_hidden, n_out, input_units, hidden_units, output_units,
                hidden_delta, output_delta, target, input_weights, hidden_weights,
                input_prev_weights, hidden_prev_weights)
end

function load_input!(net::BPNN, rng::AbstractRNG)
    for i in 2:length(net.input_units)
        net.input_units[i] = rand(rng, Float32)
    end
end

function layerforward!(l1, l2, conn, n1::Int, n2::Int)
    l1[1] = 1f0
    for j in 2:(n2 + 1)
        s = 0f0
        for k in 1:(n1 + 1)
            s += conn[k, j] * l1[k]
        end
        l2[j] = sigmoid(s)
    end
end

function output_error!(delta, target, output, nj::Int)
    errsum = 0f0
    for j in 2:(nj + 1)
        o = output[j]
        t = target[j]
        delta[j] = o * (1f0 - o) * (t - o)
        errsum += abs(delta[j])
    end
    return errsum
end

function hidden_error!(delta_h, nh::Int, delta_o, no::Int, who, hidden)
    errsum = 0f0
    for j in 2:(nh + 1)
        h = hidden[j]
        s = 0f0
        for k in 2:(no + 1)
            s += delta_o[k] * who[j, k]
        end
        delta_h[j] = h * (1f0 - h) * s
        errsum += abs(delta_h[j])
    end
    return errsum
end

function adjust_weights!(delta, ndelta::Int, ly, nly::Int, w, oldw)
    ly[1] = 1f0
    for j in 2:(ndelta + 1)
        for k in 1:(nly + 1)
            new_dw = ETA * delta[j] * ly[k] + MOMENTUM * oldw[k, j]
            w[k, j] += new_dw
            oldw[k, j] = new_dw
        end
    end
end

function cpu_reference(net::BPNN, input_weights::Matrix{Float32}, input_prev::Matrix{Float32})
    ref_weights = copy(input_weights)
    ref_prev = copy(input_prev)
    layerforward!(net.input_units, net.hidden_units, ref_weights, net.input_n, net.hidden_n)
    layerforward!(net.hidden_units, net.output_units, net.hidden_weights, net.hidden_n, net.output_n)
    output_error!(net.output_delta, net.target, net.output_units, net.output_n)
    hidden_error!(net.hidden_delta, net.hidden_n, net.output_delta, net.output_n,
                  net.hidden_weights, net.hidden_units)
    adjust_weights!(net.output_delta, net.output_n, net.hidden_units, net.hidden_n,
                    net.hidden_weights, net.hidden_prev_weights)
    adjust_weights!(net.hidden_delta, net.hidden_n, net.input_units, net.input_n,
                    ref_weights, ref_prev)
    return ref_weights
end

function gpu_train!(net::BPNN)
    CUDA.synchronize()
    start = time_ns()

    d_input = CuArray(net.input_units[2:end])
    d_weights = CuArray(net.input_weights[2:end, 2:end])
    d_bias = CuArray(net.input_weights[1, 2:end])
    d_hidden = sigmoid.(transpose(d_weights) * d_input .+ d_bias)
    hidden_host = Array(d_hidden)
    net.hidden_units[1] = 1f0
    net.hidden_units[2:end] .= hidden_host

    layerforward!(net.hidden_units, net.output_units, net.hidden_weights, net.hidden_n, net.output_n)
    output_error!(net.output_delta, net.target, net.output_units, net.output_n)
    hidden_error!(net.hidden_delta, net.hidden_n, net.output_delta, net.output_n,
                  net.hidden_weights, net.hidden_units)
    adjust_weights!(net.output_delta, net.output_n, net.hidden_units, net.hidden_n,
                    net.hidden_weights, net.hidden_prev_weights)

    d_delta = CuArray(net.hidden_delta[2:end])
    d_prev = CuArray(net.input_prev_weights[2:end, 2:end])
    d_update = ETA .* (d_input * transpose(d_delta)) .+ MOMENTUM .* d_prev
    d_weights .+= d_update
    d_bias .+= ETA .* d_delta .+ MOMENTUM .* CuArray(net.input_prev_weights[1, 2:end])
    CUDA.synchronize()

    out_weights = copy(net.input_weights)
    out_weights[2:end, 2:end] .= Array(d_weights)
    out_weights[1, 2:end] .= Array(d_bias)

    elapsed = time_ns() - start
    @printf("Device offloading time = %lf(s)\n", elapsed * 1e-9)
    return out_weights
end

function main()
    if length(ARGS) != 1
        @printf("Usage: %s <number of input nodes>\n", PROGRAM_FILE)
        return 1
    end
    layer_size = parse(Int, ARGS[1])
    if layer_size % 16 != 0
        println("The number of input nodes must be divided by 16")
        return 1
    end

    seed = 7
    @printf("Random number generator seed: %d\n", seed)
    rng = MersenneTwister(seed)
    net = create_net(layer_size, HID, OUT, rng)
    @printf("Input layer size : %d\n", layer_size)
    load_input!(net, rng)
    println("Starting training kernel")

    ref_net = deepcopy(net)
    input_weights_original = copy(net.input_weights)
    input_prev_original = copy(net.input_prev_weights)
    gpu_weights = gpu_train!(net)
    println("Performing host execution ")
    ref_weights = cpu_reference(ref_net, input_weights_original, input_prev_original)

    ok = maximum(abs.(gpu_weights .- ref_weights)) < 1f-3
    println(ok ? "PASS" : "FAIL")
    println("\nFinish the training for one iteration")
    return ok ? 0 : 1
end

exit(main())
