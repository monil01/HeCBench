using CUDA
using Printf

function lcg_value(index0::Int, size::Int)
    seed = UInt32(index0) ⊻ UInt32(size)
    seed = UInt32(26757677) * seed + UInt32(1)
    return Float32(seed) / Float32(2147483648)
end

function init_ref(size::Int)
    out = Vector{Float32}(undef, size)
    for i in 0:(size - 1)
        out[i + 1] = lcg_value(i, size)
    end
    return out
end

function sigmoid(x::Float32)
    return Float32(1) / (Float32(1) + exp(-x))
end

function elementwise_kernel!(hidden_size::Int32, mini_batch::Int32, tmp_h, tmp_i,
                             bias, linear_gates, h_out, i_out, c_in, c_out)
    index0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    num_elements = hidden_size * mini_batch
    if index0 >= num_elements
        return
    end

    batch = index0 ÷ hidden_size
    hidden_idx = index0 % hidden_size
    gate_index = hidden_idx + Int32(4) * batch * hidden_size

    @inbounds g0 = tmp_i[Int(gate_index) + 1] + tmp_h[Int(gate_index) + 1] +
                   bias[Int(hidden_idx) + 1] + bias[Int(Int32(4) * hidden_size + hidden_idx) + 1]
    @inbounds g1 = tmp_i[Int(hidden_size + gate_index) + 1] + tmp_h[Int(hidden_size + gate_index) + 1] +
                   bias[Int(hidden_size + hidden_idx) + 1] + bias[Int(Int32(5) * hidden_size + hidden_idx) + 1]
    @inbounds g2 = tmp_i[Int(Int32(2) * hidden_size + gate_index) + 1] + tmp_h[Int(Int32(2) * hidden_size + gate_index) + 1] +
                   bias[Int(Int32(2) * hidden_size + hidden_idx) + 1] + bias[Int(Int32(6) * hidden_size + hidden_idx) + 1]
    @inbounds g3 = tmp_i[Int(Int32(3) * hidden_size + gate_index) + 1] + tmp_h[Int(Int32(3) * hidden_size + gate_index) + 1] +
                   bias[Int(Int32(3) * hidden_size + hidden_idx) + 1] + bias[Int(Int32(7) * hidden_size + hidden_idx) + 1]

    @inbounds linear_gates[Int(gate_index) + 1] = g0
    @inbounds linear_gates[Int(gate_index + hidden_size) + 1] = g1
    @inbounds linear_gates[Int(gate_index + Int32(2) * hidden_size) + 1] = g2
    @inbounds linear_gates[Int(gate_index + Int32(3) * hidden_size) + 1] = g3

    in_gate = Float32(1) / (Float32(1) + exp(-g0))
    forget_gate = Float32(1) / (Float32(1) + exp(-g1))
    in_gate2 = tanh(g2)
    out_gate = Float32(1) / (Float32(1) + exp(-g3))

    @inbounds val = forget_gate * c_in[Int(index0) + 1] + in_gate * in_gate2
    @inbounds c_out[Int(index0) + 1] = val
    val = out_gate * tanh(val)
    @inbounds h_out[Int(index0) + 1] = val
    @inbounds i_out[Int(index0) + 1] = val
    return
end

function elementwise_ref!(hidden_size, mini_batch, tmp_h, tmp_i, bias,
                          linear_gates, h_out, i_out, c_in, c_out)
    num_elements = hidden_size * mini_batch
    for index0 in 0:(num_elements - 1)
        batch = index0 ÷ hidden_size
        hidden_idx = index0 % hidden_size
        gate_index = hidden_idx + 4 * batch * hidden_size
        g = ntuple(i -> begin
            k = i - 1
            tmp_i[k * hidden_size + gate_index + 1] +
            tmp_h[k * hidden_size + gate_index + 1] +
            bias[k * hidden_size + hidden_idx + 1] +
            bias[(k + 4) * hidden_size + hidden_idx + 1]
        end, 4)
        for k in 0:3
            linear_gates[gate_index + k * hidden_size + 1] = g[k + 1]
        end
        in_gate = sigmoid(g[1])
        forget_gate = sigmoid(g[2])
        in_gate2 = tanh(g[3])
        out_gate = sigmoid(g[4])
        val = forget_gate * c_in[index0 + 1] + in_gate * in_gate2
        c_out[index0 + 1] = val
        val = out_gate * tanh(val)
        h_out[index0 + 1] = val
        i_out[index0 + 1] = val
    end
end

function schedule_work(seq_length, num_layers)
    work = Tuple{Int,Int}[]
    l_start = 0
    l_end = 0
    r_start = 0
    recur_batch_size = 2
    while true
        if l_end == 0
            l_start = 0
            l_end = 1
            r_start = 0
        else
            l_start += 1
            l_end += 1
            r_start -= recur_batch_size
            if l_end > num_layers || r_start < 0
                r_start += (l_start + 1) * recur_batch_size
                l_start = 0
                l_end = 1
            end
            while r_start >= seq_length && l_end <= num_layers
                l_start += 1
                l_end += 1
                r_start -= recur_batch_size
            end
            if l_end > num_layers || r_start < 0
                break
            end
        end
        r_end = min(r_start + recur_batch_size, seq_length)
        for layer in l_start:(l_end - 1), i in r_start:(r_end - 1)
            push!(work, (layer, i))
        end
    end
    return work
end

function test_gpu(hidden_size, mini_batch, seq_length, num_layers)
    num_elements = hidden_size * mini_batch
    hc_size = (seq_length + 1) * num_layers * num_elements
    i_size = seq_length * (num_layers + 1) * num_elements
    bias_size = num_layers * hidden_size * 8
    tmp_h_size = 4 * num_layers * num_elements
    tmp_i_size = 4 * seq_length * num_elements
    lg_size = 4 * seq_length * num_layers * num_elements

    h_data = CUDA.zeros(Float32, hc_size)
    i_data = CUDA.zeros(Float32, i_size)
    c_data = CuArray(init_ref(hc_size))
    bias = CuArray(init_ref(bias_size))
    tmp_h = CuArray(init_ref(tmp_h_size))
    tmp_i = CuArray(init_ref(tmp_i_size))
    linear_gates = CUDA.zeros(Float32, lg_size)

    blocks = cld(num_elements, 256)
    total_ns = Int128(0)
    for (layer, i) in schedule_work(seq_length, num_layers)
        lg_lo = 4 * (i * num_elements + layer * seq_length * num_elements) + 1
        lg_hi = 4 * (i * num_elements + layer * seq_length * num_elements + num_elements)
        h_lo = (i + 1) * num_elements + layer * (seq_length + 1) * num_elements + 1
        h_hi = (i + 2) * num_elements + layer * (seq_length + 1) * num_elements
        io_lo = i * num_elements + (layer + 1) * seq_length * num_elements + 1
        io_hi = (i + 1) * num_elements + (layer + 1) * seq_length * num_elements
        c_in_lo = i * num_elements + layer * (seq_length + 1) * num_elements + 1
        c_in_hi = (i + 1) * num_elements + layer * (seq_length + 1) * num_elements
        c_out_lo = (i + 1) * num_elements + layer * (seq_length + 1) * num_elements + 1
        c_out_hi = (i + 2) * num_elements + layer * (seq_length + 1) * num_elements
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=256 blocks=blocks elementwise_kernel!(
            Int32(hidden_size), Int32(mini_batch),
            view(tmp_h, 4 * layer * num_elements + 1:4 * (layer + 1) * num_elements),
            view(tmp_i, 4 * i * num_elements + 1:4 * (i + 1) * num_elements),
            view(bias, 8 * layer * hidden_size + 1:8 * (layer + 1) * hidden_size),
            view(linear_gates, lg_lo:lg_hi),
            view(h_data, h_lo:h_hi),
            view(i_data, io_lo:io_hi),
            view(c_data, c_in_lo:c_in_hi),
            view(c_data, c_out_lo:c_out_hi))
        CUDA.synchronize()
        total_ns += Int128(time_ns() - start)
    end

    out_i_lo = num_layers * seq_length * num_elements + 1
    out_i_hi = (num_layers + 1) * seq_length * num_elements
    out_i = Array(view(i_data, out_i_lo:out_i_hi))
    out_h = Vector{Float32}(undef, num_elements * num_layers)
    out_c = Vector{Float32}(undef, num_elements * num_layers)
    for layer in 0:(num_layers - 1)
        lo = seq_length * num_elements + layer * (seq_length + 1) * num_elements + 1
        hi = lo + num_elements - 1
        out_h[layer * num_elements + 1:(layer + 1) * num_elements] .= Array(view(h_data, lo:hi))
        out_c[layer * num_elements + 1:(layer + 1) * num_elements] .= Array(view(c_data, lo:hi))
    end
    return out_i, out_h, out_c, total_ns
end

function test_ref(hidden_size, mini_batch, seq_length, num_layers)
    num_elements = hidden_size * mini_batch
    hc_size = (seq_length + 1) * num_layers * num_elements
    i_size = seq_length * (num_layers + 1) * num_elements
    bias_size = num_layers * hidden_size * 8
    tmp_h_size = 4 * num_layers * num_elements
    tmp_i_size = 4 * seq_length * num_elements
    lg_size = 4 * seq_length * num_layers * num_elements

    h_data = zeros(Float32, hc_size)
    i_data = zeros(Float32, i_size)
    c_data = init_ref(hc_size)
    bias = init_ref(bias_size)
    tmp_h = init_ref(tmp_h_size)
    tmp_i = init_ref(tmp_i_size)
    linear_gates = zeros(Float32, lg_size)

    for (layer, i) in schedule_work(seq_length, num_layers)
        lg_lo = 4 * (i * num_elements + layer * seq_length * num_elements) + 1
        lg_hi = 4 * (i * num_elements + layer * seq_length * num_elements + num_elements)
        h_lo = (i + 1) * num_elements + layer * (seq_length + 1) * num_elements + 1
        h_hi = (i + 2) * num_elements + layer * (seq_length + 1) * num_elements
        io_lo = i * num_elements + (layer + 1) * seq_length * num_elements + 1
        io_hi = (i + 1) * num_elements + (layer + 1) * seq_length * num_elements
        c_in_lo = i * num_elements + layer * (seq_length + 1) * num_elements + 1
        c_in_hi = (i + 1) * num_elements + layer * (seq_length + 1) * num_elements
        c_out_lo = (i + 1) * num_elements + layer * (seq_length + 1) * num_elements + 1
        c_out_hi = (i + 2) * num_elements + layer * (seq_length + 1) * num_elements
        elementwise_ref!(
            hidden_size, mini_batch,
            @view(tmp_h[4 * layer * num_elements + 1:4 * (layer + 1) * num_elements]),
            @view(tmp_i[4 * i * num_elements + 1:4 * (i + 1) * num_elements]),
            @view(bias[8 * layer * hidden_size + 1:8 * (layer + 1) * hidden_size]),
            @view(linear_gates[lg_lo:lg_hi]),
            @view(h_data[h_lo:h_hi]),
            @view(i_data[io_lo:io_hi]),
            @view(c_data[c_in_lo:c_in_hi]),
            @view(c_data[c_out_lo:c_out_hi]))
    end

    out_i_lo = num_layers * seq_length * num_elements + 1
    out_i_hi = (num_layers + 1) * seq_length * num_elements
    out_i = copy(@view i_data[out_i_lo:out_i_hi])
    out_h = Vector{Float32}(undef, num_elements * num_layers)
    out_c = Vector{Float32}(undef, num_elements * num_layers)
    for layer in 0:(num_layers - 1)
        lo = seq_length * num_elements + layer * (seq_length + 1) * num_elements + 1
        hi = lo + num_elements - 1
        out_h[layer * num_elements + 1:(layer + 1) * num_elements] .= @view h_data[lo:hi]
        out_c[layer * num_elements + 1:(layer + 1) * num_elements] .= @view c_data[lo:hi]
    end
    return out_i, out_h, out_c
end

function main(args)
    if length(args) == 5
        seq_length, num_layers, hidden_size, mini_batch, num_runs = parse.(Int, args)
    elseif isempty(args)
        println("Running with default settings")
        seq_length, num_layers, hidden_size, mini_batch, num_runs = 100, 4, 512, 64, 1
    else
        println("Usage: main.jl <seqLength> <numLayers> <hiddenSize> <miniBatch> <repeat>")
        return 1
    end

    @printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n",
            seq_length, num_layers, hidden_size, mini_batch)

    num_elements = hidden_size * mini_batch
    test_output_i = Float32[]
    test_output_h = Float32[]
    test_output_c = Float32[]
    time_ns_total = Int128(0)

    for _ in 1:num_runs
        test_output_i, test_output_h, test_output_c, elapsed = test_gpu(hidden_size, mini_batch, seq_length, num_layers)
        time_ns_total += elapsed
    end
    ref_i, ref_h, ref_c = test_ref(hidden_size, mini_batch, seq_length, num_layers)

    @printf("Average kernel execution time: %f (s)\n", Float64(time_ns_total) * 1e-9 / num_runs)

    error = 0
    for m in 0:(mini_batch - 1)
        for j in 0:(seq_length - 1), i in 0:(hidden_size - 1)
            idx = j * num_elements + m * hidden_size + i + 1
            if abs(test_output_i[idx] - ref_i[idx]) > Float32(1.0f-4)
                error += 1
            end
        end
        for j in 0:(num_layers - 1), i in 0:(hidden_size - 1)
            idx = j * num_elements + m * hidden_size + i + 1
            if abs(test_output_h[idx] - ref_h[idx]) > Float32(1.0f-4)
                error += 1
            end
            if abs(test_output_c[idx] - ref_c[idx]) > Float32(1.0f-4)
                error += 1
            end
        end
    end
    println(error == 0 ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
