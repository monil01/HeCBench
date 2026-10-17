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

