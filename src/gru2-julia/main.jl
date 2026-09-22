using CUDA
using Printf
using Random

const THREADS = 128

@inline sigmoidf(x) = 1.0f0 / (1.0f0 + exp(-x))

@inline function wih_offset(layer, gate, hidden_idx, input_idx, hidden_size, max_input_size)
    return (((layer * 3 + gate) * hidden_size + hidden_idx) * max_input_size + input_idx) + 1
end

@inline function whh_offset(layer, gate, hidden_idx, prev_hidden_idx, hidden_size)
    return (((layer * 3 + gate) * hidden_size + hidden_idx) * hidden_size + prev_hidden_idx) + 1
end

@inline function bias_offset(layer, gate, hidden_idx, hidden_size)
    return ((layer * 3 + gate) * hidden_size + hidden_idx) + 1
end

function fused_gru_layer_kernel!(x, h_prev, w_ih, w_hh, b_ih, b_hh,
                                 layer_out, h_next, output_t,
                                 layer::Int32, batch_size::Int32,
                                 input_size::Int32, hidden_size::Int32,
                                 max_input_size::Int32, write_output::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = batch_size * hidden_size
    if idx >= total
        return
    end

    b = idx ÷ hidden_size
    h = idx - b * hidden_size

    input_gate0 = 0.0f0
    input_gate1 = 0.0f0
    input_gate2 = 0.0f0
    hidden_gate0 = 0.0f0
    hidden_gate1 = 0.0f0
    hidden_gate2 = 0.0f0

    @inbounds begin
        for k in Int32(0):(input_size - Int32(1))
            xv = x[b * input_size + k + Int32(1)]
            input_gate0 += xv * w_ih[wih_offset(layer, Int32(0), h, k, hidden_size, max_input_size)]
            input_gate1 += xv * w_ih[wih_offset(layer, Int32(1), h, k, hidden_size, max_input_size)]
            input_gate2 += xv * w_ih[wih_offset(layer, Int32(2), h, k, hidden_size, max_input_size)]
        end
        input_gate0 += b_ih[bias_offset(layer, Int32(0), h, hidden_size)]
        input_gate1 += b_ih[bias_offset(layer, Int32(1), h, hidden_size)]
        input_gate2 += b_ih[bias_offset(layer, Int32(2), h, hidden_size)]

        hbase = b * hidden_size
        for k in Int32(0):(hidden_size - Int32(1))
            hv = h_prev[hbase + k + Int32(1)]
            hidden_gate0 += hv * w_hh[whh_offset(layer, Int32(0), h, k, hidden_size)]
            hidden_gate1 += hv * w_hh[whh_offset(layer, Int32(1), h, k, hidden_size)]
            hidden_gate2 += hv * w_hh[whh_offset(layer, Int32(2), h, k, hidden_size)]
        end
        hidden_gate0 += b_hh[bias_offset(layer, Int32(0), h, hidden_size)]
        hidden_gate1 += b_hh[bias_offset(layer, Int32(1), h, hidden_size)]
        hidden_gate2 += b_hh[bias_offset(layer, Int32(2), h, hidden_size)]

        hx = h_prev[idx + Int32(1)]
        r = sigmoidf(input_gate0 + hidden_gate0)
        z = sigmoidf(input_gate1 + hidden_gate1)
        n = tanh(input_gate2 + r * hidden_gate2)
        ht = n + z * (hx - n)
        h_next[idx + Int32(1)] = ht
        layer_out[idx + Int32(1)] = ht
        if write_output != Int32(0)
            output_t[idx + Int32(1)] = ht
        end
    end
    return
end

function gru!(d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0, d_output,
              d_hidden, d_hidden_a, d_layer_output_a, d_layer_output_b,
              seq_len, batch_size, input_size, hidden_size, num_layers,
              max_input_size)
    hidden_read = d_h0
    hidden_buffers = (d_hidden, d_hidden_a)
    hidden_write_idx = 1
    blocks = cld(batch_size * hidden_size, THREADS)

    for t in 0:(seq_len - 1)
        hidden_write = hidden_buffers[hidden_write_idx]
        current_input = @view d_input[(t * batch_size * input_size + 1):((t + 1) * batch_size * input_size)]
        current_input_size = input_size
        output_t = @view d_output[(t * batch_size * hidden_size + 1):((t + 1) * batch_size * hidden_size)]

        for layer in 0:(num_layers - 1)
            hidden_offset = layer * batch_size * hidden_size
            layer_hidden_read = @view hidden_read[(hidden_offset + 1):(hidden_offset + batch_size * hidden_size)]
            layer_hidden_write = @view hidden_write[(hidden_offset + 1):(hidden_offset + batch_size * hidden_size)]
            current_output = iseven(layer) ? d_layer_output_a : d_layer_output_b
            if layer > 0
                current_input = isodd(layer) ? d_layer_output_a : d_layer_output_b
                current_input_size = hidden_size
            end
            write_output = layer == num_layers - 1 ? Int32(1) : Int32(0)
            @cuda threads=THREADS blocks=blocks fused_gru_layer_kernel!(
                current_input, layer_hidden_read, d_w_ih, d_w_hh, d_b_ih, d_b_hh,
                current_output, layer_hidden_write, output_t, Int32(layer),
                Int32(batch_size), Int32(current_input_size), Int32(hidden_size),
                Int32(max_input_size), write_output)
        end
        hidden_read = hidden_write
        hidden_write_idx = 3 - hidden_write_idx
    end
    return hidden_read
end

function gru_reference(input, w_ih, w_hh, b_ih, b_hh, h0,
                       seq_len, batch_size, input_size, hidden_size,
                       num_layers, max_input_size)
    output = Vector{Float32}(undef, seq_len * batch_size * hidden_size)
    hn = copy(h0)
    layer_in = Vector{Float32}(undef, batch_size * max_input_size)
    layer_out = Vector{Float32}(undef, batch_size * hidden_size)

    @inbounds for t in 0:(seq_len - 1)
        current_input = view(input, (t * batch_size * input_size + 1):((t + 1) * batch_size * input_size))
        current_input_size = input_size
        for layer in 0:(num_layers - 1)
            if layer > 0
                current_input = view(layer_in, 1:(batch_size * hidden_size))
                current_input_size = hidden_size
            end
            for b in 0:(batch_size - 1), h in 0:(hidden_size - 1)
                ig0 = b_ih[bias_offset(layer, 0, h, hidden_size)]
                ig1 = b_ih[bias_offset(layer, 1, h, hidden_size)]
                ig2 = b_ih[bias_offset(layer, 2, h, hidden_size)]
                for k in 0:(current_input_size - 1)
                    xv = current_input[b * current_input_size + k + 1]
                    ig0 += xv * w_ih[wih_offset(layer, 0, h, k, hidden_size, max_input_size)]
                    ig1 += xv * w_ih[wih_offset(layer, 1, h, k, hidden_size, max_input_size)]
                    ig2 += xv * w_ih[wih_offset(layer, 2, h, k, hidden_size, max_input_size)]
                end

                hprev_base = (layer * batch_size + b) * hidden_size
                hg0 = b_hh[bias_offset(layer, 0, h, hidden_size)]
                hg1 = b_hh[bias_offset(layer, 1, h, hidden_size)]
                hg2 = b_hh[bias_offset(layer, 2, h, hidden_size)]
                for k in 0:(hidden_size - 1)
                    hv = hn[hprev_base + k + 1]
                    hg0 += hv * w_hh[whh_offset(layer, 0, h, k, hidden_size)]
                    hg1 += hv * w_hh[whh_offset(layer, 1, h, k, hidden_size)]
                    hg2 += hv * w_hh[whh_offset(layer, 2, h, k, hidden_size)]
                end

                r = sigmoidf(ig0 + hg0)
                z = sigmoidf(ig1 + hg1)
                n = tanh(ig2 + r * hg2)
                layer_out[b * hidden_size + h + 1] = n + z * (hn[hprev_base + h + 1] - n)
            end
            for b in 0:(batch_size - 1), h in 0:(hidden_size - 1)
                v = layer_out[b * hidden_size + h + 1]
                hn[(layer * batch_size + b) * hidden_size + h + 1] = v
                layer_in[b * hidden_size + h + 1] = v
            end
        end
        copyto!(output, t * batch_size * hidden_size + 1, layer_out, 1, batch_size * hidden_size)
    end
    return output, hn
end

function main(args)
    if length(args) != 6
        println("Usage: main.jl <seq_len> <batch_size> <input_size> <hidden_size> <num_layers> <repeat>")
        return 1
    end
    seq_len, batch_size, input_size, hidden_size, num_layers, repeat = parse.(Int, args)
    if any(x -> x <= 0, (seq_len, batch_size, input_size, hidden_size, num_layers, repeat))
        println(stderr, "All arguments must be positive integers.")
        return 1
    end

    max_input_size = max(input_size, hidden_size)
    input_elements = seq_len * batch_size * input_size
    output_elements = seq_len * batch_size * hidden_size
    hidden_state_elements = num_layers * batch_size * hidden_size
    w_ih_elements = num_layers * 3 * hidden_size * max_input_size
    w_hh_elements = num_layers * 3 * hidden_size * hidden_size
    bias_elements = num_layers * 3 * hidden_size

    rng = MersenneTwister(123)
    h_input = rand(rng, Float32, input_elements) .* 0.4f0 .- 0.2f0
    h_w_ih = rand(rng, Float32, w_ih_elements) .* 0.4f0 .- 0.2f0
    h_w_hh = rand(rng, Float32, w_hh_elements) .* 0.4f0 .- 0.2f0
    h_b_ih = rand(rng, Float32, bias_elements) .* 0.4f0 .- 0.2f0
    h_b_hh = rand(rng, Float32, bias_elements) .* 0.4f0 .- 0.2f0
    h_h0 = rand(rng, Float32, hidden_state_elements) .* 0.4f0 .- 0.2f0

    h_output_ref, h_hidden_ref = gru_reference(h_input, h_w_ih, h_w_hh, h_b_ih, h_b_hh,
                                               h_h0, seq_len, batch_size, input_size,
                                               hidden_size, num_layers, max_input_size)

    d_input = CuArray(h_input)
    d_w_ih = CuArray(h_w_ih)
    d_w_hh = CuArray(h_w_hh)
    d_b_ih = CuArray(h_b_ih)
    d_b_hh = CuArray(h_b_hh)
    d_h0 = CuArray(h_h0)
    d_output = CUDA.zeros(Float32, output_elements)
    d_hidden = CUDA.zeros(Float32, hidden_state_elements)
    d_hidden_a = CUDA.zeros(Float32, hidden_state_elements)
    d_layer_output_a = CUDA.zeros(Float32, batch_size * hidden_size)
    d_layer_output_b = CUDA.zeros(Float32, batch_size * hidden_size)

    d_final_hidden = gru!(d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0, d_output,
                          d_hidden, d_hidden_a, d_layer_output_a, d_layer_output_b,
                          seq_len, batch_size, input_size, hidden_size, num_layers,
                          max_input_size)
    CUDA.synchronize()
    h_output = Array(d_output)
    h_hidden = Array(d_final_hidden)
    max_output_error = maximum(abs.(h_output .- h_output_ref))
    max_hidden_error = maximum(abs.(h_hidden .- h_hidden_ref))

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        d_final_hidden = gru!(d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0, d_output,
                              d_hidden, d_hidden_a, d_layer_output_a, d_layer_output_b,
                              seq_len, batch_size, input_size, hidden_size, num_layers,
                              max_input_size)
    end
    CUDA.synchronize()
    elapsed_ns = time_ns() - t0

    @printf("Average execution time of multi_layer_gru: %f (us)\n", elapsed_ns * 1.0e-3 / repeat)
    @printf("max_output_error: %.8e\n", max_output_error)
    @printf("max_hidden_error: %.8e\n", max_hidden_error)
    println(max_output_error < 1.0f-4 && max_hidden_error < 1.0f-4 ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
