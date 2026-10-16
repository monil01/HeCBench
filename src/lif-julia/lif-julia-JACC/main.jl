using CUDA,JACC
using Printf

const RAND_MAX_C = Float32(2147483647)

function c_srand(seed::UInt32)
    ccall(:srand, Cvoid, (Cuint,), seed)
end

function c_rand()
    return ccall(:rand, Cint, ())
end

function lif_kernel!(i,num_neurons::Int32, neurons_per_item::Int32, dt::Float32,
                     encode_result, voltage_array, reftime_array,
                     tau_rc::Float32, tau_ref::Float32,
                     bias, gain, spikes)
    i0=i-1
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
    peritem=length(args)>=1 ? parse(Int,args[1]) : 64
    items=length(args)>=2 ? parse(Int,args[2]) : 64
    steps=length(args)>=3 ? parse(Int,args[3]) : 10
    n=peritem*items;dt=0.1f0;tau_rc=10f0;tau_ref=2f0
    encode,bias,gain,voltage,reftime=init_inputs(peritem,items)
    refv=copy(voltage);refr=copy(reftime);refs=zeros(Float32,n)
    for _ in 1:steps
        reference!(n,peritem,dt,encode,refv,refr,tau_rc,tau_ref,bias,gain,refs)
    end
    expected=vcat(refv,refr,refs)
    spec=JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    de=CuArray(encode);db=CuArray(bias);dg=CuArray(gain);dv=CuArray(voltage);dr=CuArray(reftime);ds=CUDA.zeros(Float32,n)
    JACC.parallel_for(spec,n,lif_kernel!,Int32(n),Int32(peritem),dt,de,dv,dr,tau_rc,tau_ref,db,dg,ds)
    CUDA.synchronize()
    for trial in 1:5
        copyto!(dv,voltage);copyto!(dr,reftime);fill!(ds,0f0);CUDA.synchronize();start=time_ns()
        for _ in 1:steps
            JACC.parallel_for(spec,n,lif_kernel!,Int32(n),Int32(peritem),dt,de,dv,dr,tau_rc,tau_ref,db,dg,ds)
        end
        CUDA.synchronize();seconds=(time_ns()-start)*1e-9/steps
        result=vcat(Array(dv),Array(dr),Array(ds));error=maximum(abs.(result.-expected));correct=all(isfinite,result)&&all(abs.(result.-expected).<= 1f-4 .+ 1f-5 .* abs.(expected))
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,result);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,
         ",\"checked_values\":",3n,",\"max_abs_error\":",error,",\"timing_region\":\"synchronized_neuron_steps_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":",steps,
         ",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float32\",\"count\":",3n,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))
