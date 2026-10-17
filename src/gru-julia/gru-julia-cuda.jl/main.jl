using CUDA
using Printf
using Random

const THREADS = 256

@inline function sigmoid_f32(x::Float32)
    return 1.0f0 / (1.0f0 + exp(-x))
end

function gru_cell_forward!(input, hidden, bias1, bias2, hx, hy, storage,
                           hsz::Int32, total_elements::Int32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int32(blockDim().x * gridDim().x)
    while i <= total_elements
        linear = i - Int32(1)
        batch_idx = linear ÷ hsz
        hidden_idx = linear - batch_idx * hsz
        offset = batch_idx * Int32(3) * hsz + hidden_idx + Int32(1)

        ir = @inbounds input[offset]
        ii = @inbounds input[offset + hsz]
        inn = @inbounds input[offset + Int32(2) * hsz]
        hr = @inbounds hidden[offset]
        hi = @inbounds hidden[offset + hsz]
        hn = @inbounds hidden[offset + Int32(2) * hsz]
        hxv = @inbounds hx[i]

        bidx = hidden_idx + Int32(1)
        b1r = @inbounds bias1[bidx]
        b1i = @inbounds bias1[bidx + hsz]
        b1n = @inbounds bias1[bidx + Int32(2) * hsz]
        b2r = @inbounds bias2[bidx]
        b2i = @inbounds bias2[bidx + hsz]
        b2n = @inbounds bias2[bidx + Int32(2) * hsz]

        rg = sigmoid_f32(Float32(ir) + Float32(hr) + Float32(b1r) + Float32(b2r))
        ig = sigmoid_f32(Float32(ii) + Float32(hi) + Float32(b1i) + Float32(b2i))
        ng = tanh(Float32(inn) + Float32(b1n) + rg * (Float32(hn) + Float32(b2n)))
        @inbounds hy[i] = Float16(ng + ig * (Float32(hxv) - ng))

        store_offset = batch_idx * Int32(5) * hsz + hidden_idx + Int32(1)
        @inbounds storage[store_offset] = Float16(rg)
        @inbounds storage[store_offset + hsz] = Float16(ig)
        @inbounds storage[store_offset + Int32(2) * hsz] = Float16(ng)
        @inbounds storage[store_offset + Int32(3) * hsz] = hxv
        @inbounds storage[store_offset + Int32(4) * hsz] = Float16(Float32(hn) + Float32(b2n))
        i += stride
    end
    return
end

function reference(input, hidden, bias1, bias2, hx, hsz::Int, total_elements::Int)
    hy = Vector{Float16}(undef, total_elements)
    storage = Vector{Float16}(undef, 5 * total_elements)
    @inbounds for linear in 0:total_elements-1
        batch_idx = linear ÷ hsz
        hidden_idx = linear % hsz
        offset = batch_idx * 3 * hsz + hidden_idx + 1
        bidx = hidden_idx + 1

        rg = sigmoid_f32(Float32(input[offset]) + Float32(hidden[offset]) +
                         Float32(bias1[bidx]) + Float32(bias2[bidx]))
        ig = sigmoid_f32(Float32(input[offset + hsz]) + Float32(hidden[offset + hsz]) +
                         Float32(bias1[bidx + hsz]) + Float32(bias2[bidx + hsz]))
        ng = tanh(Float32(input[offset + 2 * hsz]) + Float32(bias1[bidx + 2 * hsz]) +
                  rg * (Float32(hidden[offset + 2 * hsz]) + Float32(bias2[bidx + 2 * hsz])))
        hy[linear + 1] = Float16(ng + ig * (Float32(hx[linear + 1]) - ng))

        store_offset = batch_idx * 5 * hsz + hidden_idx + 1
        storage[store_offset] = Float16(rg)
        storage[store_offset + hsz] = Float16(ig)
        storage[store_offset + 2 * hsz] = Float16(ng)
        storage[store_offset + 3 * hsz] = hx[linear + 1]
        storage[store_offset + 4 * hsz] = Float16(Float32(hidden[offset + 2 * hsz]) +
                                                  Float32(bias2[bidx + 2 * hsz]))
    end
    return hy, storage
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of sequences> <hidden size> <repeat>")
        return 1
    end
    vsz = parse(Int, args[1])
    hsz = parse(Int, args[2])
    repeat = parse(Int, args[3])
    total_elements = vsz * hsz

    rng = MersenneTwister(123)
    input = Float16.(rand(rng, Float32, 3 * total_elements) .* 4.0f0 .- 2.0f0)
    hidden = Float16.(rand(rng, Float32, 3 * total_elements) .* 4.0f0 .- 2.0f0)
    bias1 = Float16.(rand(rng, Float32, 3 * hsz) .* 4.0f0 .- 2.0f0)
    bias2 = Float16.(rand(rng, Float32, 3 * hsz) .* 4.0f0 .- 2.0f0)
    hx = Float16.(rand(rng, Float32, total_elements) .* 4.0f0 .- 2.0f0)

    d_input = CuArray(input)
    d_hidden = CuArray(hidden)
    d_bias1 = CuArray(bias1)
    d_bias2 = CuArray(bias2)
    d_hx = CuArray(hx)
    d_hy = CUDA.zeros(Float16, total_elements)
    d_storage = CUDA.zeros(Float16, 5 * total_elements)
    blocks = max(cld(total_elements, THREADS), 1)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=blocks gru_cell_forward!(
            d_input, d_hidden, d_bias1, d_bias2, d_hx, d_hy, d_storage,
            Int32(hsz), Int32(total_elements))
    end
    CUDA.synchronize()
    @printf("Average execution time of gru_cell_forward: %f (us)\n",
            (time_ns() - t0) * 1.0e-3 / repeat)

    hy_ref, storage_ref = reference(input, hidden, bias1, bias2, hx, hsz, total_elements)
    hy = Array(d_hy)
    storage = Array(d_storage)
    ok = all(abs.(Float32.(hy) .- Float32.(hy_ref)) .<= 1.0f-3) &&
         all(abs.(Float32.(storage) .- Float32.(storage_ref)) .<= 1.0f-3)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
