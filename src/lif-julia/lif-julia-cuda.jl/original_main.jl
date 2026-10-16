using CUDA
using Printf

const RAND_MAX_C = Float32(2147483647)

function c_srand(seed::UInt32)
    ccall(:srand, Cvoid, (Cuint,), seed)
end

function c_rand()
    return ccall(:rand, Cint, ())
end

function lif_kernel!(num_neurons::Int32, neurons_per_item::Int32, dt::Float32,
                     encode_result, voltage_array, reftime_array,
                     tau_rc::Float32, tau_ref::Float32,
                     bias, gain, spikes)
    i0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i0 < num_neurons
        neuron_index = i0 % neurons_per_item
        item_index = i0 ÷ neurons_per_item
        pos = i0 + Int32(1)

        @inbounds begin
            voltage = voltage_array[pos]
            ref_time = reftime_array[pos]
            current = bias[neuron_index + Int32(1)] +
                gain[neuron_index + Int32(1)] * encode_result[item_index + Int32(1)]
            dV = -expm1(-dt / tau_rc) * (current - voltage)
            voltage = max(voltage + dV, 0.0f0)

            ref_time -= dt
            mult = ref_time
            mult *= -1.0f0 / dt
            mult += 1.0f0
            mult = min(mult, 1.0f0)
            mult = max(mult, 0.0f0)

            voltage *= mult

            if voltage > 1.0f0
                spike = 1.0f0 / dt
                ref_time = tau_ref + dt * (1.0f0 - (voltage - 1.0f0) / dV)
                voltage = 0.0f0
            else
                spike = 0.0f0
            end

            reftime_array[pos] = ref_time
            voltage_array[pos] = voltage
            spikes[pos] = spike
        end
    end
    return
end

function reference!(num_neurons::Int, neurons_per_item::Int, dt::Float32,
                    encode_result, voltage_array, reftime_array,
                    tau_rc::Float32, tau_ref::Float32,
                    bias, gain, spikes)
    for i0 in 0:num_neurons-1
        neuron_index = i0 % neurons_per_item
        item_index = i0 ÷ neurons_per_item
        pos = i0 + 1

        voltage = voltage_array[pos]
        ref_time = reftime_array[pos]
        current = bias[neuron_index + 1] + gain[neuron_index + 1] * encode_result[item_index + 1]

        dV = -expm1(-dt / tau_rc) * (current - voltage)
        voltage = max(voltage + dV, 0.0f0)

        ref_time -= dt
        mult = ref_time
        mult *= -1.0f0 / dt
        mult += 1.0f0
        mult = min(mult, 1.0f0)
        mult = max(mult, 0.0f0)

        voltage *= mult

        if voltage > 1.0f0
            spike = 1.0f0 / dt
            ref_time = tau_ref + dt * (1.0f0 - (voltage - 1.0f0) / dV)
            voltage = 0.0f0
        else
            spike = 0.0f0
        end

        reftime_array[pos] = ref_time
        voltage_array[pos] = voltage
        spikes[pos] = spike
    end
end

function init_inputs(neurons_per_item::Int, num_items::Int)
    num_neurons = neurons_per_item * num_items
    c_srand(UInt32(123))

    encode_result = Vector{Float32}(undef, num_items)
    bias = Vector{Float32}(undef, neurons_per_item)
    gain = Vector{Float32}(undef, neurons_per_item)
    voltage = Vector{Float32}(undef, num_neurons)
    reftime = Vector{Float32}(undef, num_neurons)

    for i in eachindex(encode_result)
        encode_result[i] = Float32(c_rand()) / RAND_MAX_C
    end
    for i in 1:num_neurons
        voltage[i] = 1.0f0 + Float32(c_rand()) / RAND_MAX_C
        reftime[i] = Float32(c_rand() % 5) / 10.0f0
    end
    for i in eachindex(bias)
        bias[i] = Float32(c_rand()) / RAND_MAX_C
        gain[i] = Float32(c_rand()) / RAND_MAX_C + 0.5f0
    end

    return encode_result, bias, gain, voltage, reftime
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <neurons per item> <num_items> <num_steps>")
        return 1
    end

    neurons_per_item = parse(Int, args[1])
    num_items = parse(Int, args[2])
    num_steps = parse(Int, args[3])
    num_neurons = neurons_per_item * num_items

    dt = 0.1f0
    tau_rc = 10.0f0
    tau_ref = 2.0f0

    encode_result, bias, gain, voltage, reftime = init_inputs(neurons_per_item, num_items)
    voltage_host = copy(voltage)
    reftime_host = copy(reftime)
    spikes_host = zeros(Float32, num_neurons)

    d_encode_result = CuArray(encode_result)
    d_bias = CuArray(bias)
    d_gain = CuArray(gain)
    d_voltage = CuArray(voltage)
    d_reftime = CuArray(reftime)
    d_spikes = CUDA.zeros(Float32, num_neurons)

    blocks = cld(num_neurons, 256)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:num_steps
        @cuda threads=256 blocks=blocks lif_kernel!(
            Int32(num_neurons), Int32(neurons_per_item), dt,
            d_encode_result, d_voltage, d_reftime,
            tau_rc, tau_ref, d_bias, d_gain, d_spikes)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1.0e-3 / num_steps
    @printf("Average kernel execution time: %f (us)\n", elapsed_us)

    spikes = Array(d_spikes)
    voltage = Array(d_voltage)
    reftime = Array(d_reftime)

    for _ in 1:num_steps
        reference!(num_neurons, neurons_per_item, dt, encode_result,
                   voltage_host, reftime_host, tau_rc, tau_ref,
                   bias, gain, spikes_host)
    end

    reftime_on_spike = Float32[]
    reftime_on_spike_host = Float32[]
    for i in 1:num_neurons
        if spikes[i] == 1.0f0 / dt
            push!(reftime_on_spike, reftime[i])
        end
        if spikes_host[i] == 1.0f0 / dt
            push!(reftime_on_spike_host, reftime_host[i])
        end
    end

    @printf("Number of spikes on host and device: %zu %zu\n",
            length(reftime_on_spike_host), length(reftime_on_spike))
    n = min(length(reftime_on_spike), length(reftime_on_spike_host))
    ok = true
    for i in 1:n
        if abs(reftime_on_spike[i] - reftime_on_spike_host[i]) > 0.1f0
            @printf("%f %f\n", reftime_on_spike[i], reftime_on_spike_host[i])
            ok = false
            break
        end
    end

    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
