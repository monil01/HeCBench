@inline function pointwise_body!(index0::Int32,hidden_size::Int32, mini_batch::Int32, tmp_h, tmp_i,
                             bias, linear_gates, h_out, i_out, c_in, c_out)
    num_elements = hidden_size * mini_batch
    if index0 >= num_elements
        return
    end

    batch = index0 ÷ hidden_size
    hidden_idx = index0 % hidden_size
    gate_index = hidden_idx + Int32(4) * batch * hidden_size

    @inbounds g0 = (tmp_i[Int(gate_index) + 1] + tmp_h[Int(gate_index) + 1]) + (bias[Int(hidden_idx) + 1] + bias[Int(Int32(4) * hidden_size + hidden_idx) + 1])
    @inbounds g1 = (tmp_i[Int(hidden_size + gate_index) + 1] + tmp_h[Int(hidden_size + gate_index) + 1]) + (bias[Int(hidden_size + hidden_idx) + 1] + bias[Int(Int32(5) * hidden_size + hidden_idx) + 1])
    @inbounds g2 = (tmp_i[Int(Int32(2) * hidden_size + gate_index) + 1] + tmp_h[Int(Int32(2) * hidden_size + gate_index) + 1]) + (bias[Int(Int32(2) * hidden_size + hidden_idx) + 1] + bias[Int(Int32(6) * hidden_size + hidden_idx) + 1])
    @inbounds g3 = (tmp_i[Int(Int32(3) * hidden_size + gate_index) + 1] + tmp_h[Int(Int32(3) * hidden_size + gate_index) + 1]) + (bias[Int(Int32(3) * hidden_size + hidden_idx) + 1] + bias[Int(Int32(7) * hidden_size + hidden_idx) + 1])

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

