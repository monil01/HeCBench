using CUDA
using Printf
using Random

function zoom_in_kernel!(input, output,
                         input_h::Int32, input_w::Int32,
                         output_h::Int32, output_w::Int32,
                         pitch::Int32,
                         out_h_start::Int32, out_h_end::Int32,
                         out_w_start::Int32, out_w_end::Int32)
    oh = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    ow = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    b = blockIdx().z - Int32(1)
    img_start = b * pitch

    if oh < output_h && ow < output_w &&
       oh >= out_h_start && oh < out_h_end &&
       ow >= out_w_start && ow < out_w_end
        ratio_h = Float32(input_h) / Float32(output_h)
        ratio_w = Float32(input_w) / Float32(output_w)
        start_h = Int32(floor(Float32(oh) * ratio_h))
        end_h = Int32(ceil(Float32(oh + Int32(1)) * ratio_h))
        start_w = Int32(floor(Float32(ow) * ratio_w))
        end_w = Int32(ceil(Float32(ow + Int32(1)) * ratio_w))
        del_h = end_h - start_h
        del_w = end_w - start_w
        s = 0.0f0
        for i in Int32(0):del_h-Int32(1)
            src_row = start_h + i
            if src_row < input_h
                for j in Int32(0):del_w-Int32(1)
                    src_col = start_w + j
                    if src_col < input_w
                        @inbounds s += input[img_start + src_row * input_w + src_col + Int32(1)]
                    end
                end
            end
        end
        @inbounds output[img_start + (oh - out_h_start) * input_w + (ow - out_w_start) + Int32(1)] =
            s / Float32(del_h * del_w)
    end
    return
end

function zoom_out_kernel!(input, output,
                          input_h::Int32, input_w::Int32,
                          output_h::Int32, output_w::Int32,
                          pitch::Int32,
                          out_h_start::Int32, out_h_end::Int32,
                          out_w_start::Int32, out_w_end::Int32)
    oh = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    ow = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    b = blockIdx().z - Int32(1)
    img_start = b * pitch

    if oh < output_h && ow < output_w
        ratio_h = Float32(input_h) / Float32(output_h)
        ratio_w = Float32(input_w) / Float32(output_w)
        start_h = Int32(floor(Float32(oh) * ratio_h))
        end_h = Int32(ceil(Float32(oh + Int32(1)) * ratio_h))
        start_w = Int32(floor(Float32(ow) * ratio_w))
        end_w = Int32(ceil(Float32(ow + Int32(1)) * ratio_w))
        del_h = end_h - start_h
        del_w = end_w - start_w
        s = 0.0f0
        for i in Int32(0):del_h-Int32(1)
            src_row = start_h + i
            if src_row < input_h
                for j in Int32(0):del_w-Int32(1)
                    src_col = start_w + j
                    if src_col < input_w
                        @inbounds s += input[img_start + src_row * input_w + src_col + Int32(1)]
                    end
                end
            end
        end
        @inbounds output[img_start + (oh + out_h_start) * input_w + (ow + out_w_start) + Int32(1)] =
            s / Float32(del_h * del_w)
    end
    return
end

function zoom_out_edge_pad_kernel!(output, height::Int32, width::Int32, pitch::Int32,
                                   no_h_start::Int32, no_w_start::Int32,
                                   no_h_end::Int32, no_w_end::Int32)
    oh = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    ow = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    b = blockIdx().z - Int32(1)
    img_start = b * pitch

    if oh < height && ow < width
        loc = img_start + oh * width + ow + Int32(1)
        src = loc
        if oh < no_h_start && ow >= no_w_start && ow < no_w_end
            src = img_start + no_h_start * width + ow + Int32(1)
        elseif oh >= no_h_end && ow >= no_w_start && ow < no_w_end
            src = img_start + (no_h_end - Int32(1)) * width + ow + Int32(1)
        elseif ow < no_w_start && oh >= no_h_start && oh < no_h_end
            src = img_start + oh * width + no_w_start + Int32(1)
        elseif ow >= no_w_end && oh >= no_h_start && oh < no_h_end
            src = img_start + oh * width + (no_w_end - Int32(1)) + Int32(1)
        elseif oh < no_h_start && ow < no_w_start
            src = img_start + no_h_start * width + no_w_start + Int32(1)
        elseif oh < no_h_start && ow >= no_w_end
            src = img_start + no_h_start * width + (no_w_end - Int32(1)) + Int32(1)
        elseif oh >= no_h_end && ow < no_w_start
            src = img_start + (no_h_end - Int32(1)) * width + no_w_start + Int32(1)
        elseif oh >= no_h_end && ow >= no_w_end
            src = img_start + (no_h_end - Int32(1)) * width + (no_w_end - Int32(1)) + Int32(1)
        end
        @inbounds output[loc] = output[src]
    end
    return
end

function zoom_in_ref!(input, output, h, w, ho, wo, pitch, hs, he, ws, we, batch)
    ratio_h = h / ho
    ratio_w = w / wo
    for b in 0:batch-1, oh in 0:ho-1, ow in 0:wo-1
        if oh < hs || oh >= he || ow < ws || ow >= we
            continue
        end
        start_h = floor(Int, oh * ratio_h)
        end_h = ceil(Int, (oh + 1) * ratio_h)
        start_w = floor(Int, ow * ratio_w)
        end_w = ceil(Int, (ow + 1) * ratio_w)
        s = 0.0f0
        for ih in start_h:end_h-1, iw in start_w:end_w-1
            if ih < h && iw < w
                s += input[b * pitch + ih * w + iw + 1]
            end
        end
        output[b * pitch + (oh - hs) * w + (ow - ws) + 1] =
            s / Float32((end_h - start_h) * (end_w - start_w))
    end
end

function zoom_out_ref!(input, output, h, w, ho, wo, pitch, hs, he, ws, we, batch)
    ratio_h = h / ho
    ratio_w = w / wo
    for b in 0:batch-1, oh in 0:ho-1, ow in 0:wo-1
        start_h = floor(Int, oh * ratio_h)
        end_h = ceil(Int, (oh + 1) * ratio_h)
        start_w = floor(Int, ow * ratio_w)
        end_w = ceil(Int, (ow + 1) * ratio_w)
        s = 0.0f0
        for ih in start_h:end_h-1, iw in start_w:end_w-1
            if ih < h && iw < w
                s += input[b * pitch + ih * w + iw + 1]
            end
        end
        output[b * pitch + (oh + hs) * w + (ow + ws) + 1] =
            s / Float32((end_h - start_h) * (end_w - start_w))
    end
end

function edge_pad_ref!(output, h, w, pitch, hs, ws, he, we, batch)
    for b in 0:batch-1, oh in 0:h-1, ow in 0:w-1
        loc = b * pitch + oh * w + ow + 1
        src = loc
        if oh < hs && ow >= ws && ow < we
            src = b * pitch + hs * w + ow + 1
        elseif oh >= he && ow >= ws && ow < we
            src = b * pitch + (he - 1) * w + ow + 1
        elseif ow < ws && oh >= hs && oh < he
            src = b * pitch + oh * w + ws + 1
        elseif ow >= we && oh >= hs && oh < he
            src = b * pitch + oh * w + (we - 1) + 1
        elseif oh < hs && ow < ws
            src = b * pitch + hs * w + ws + 1
        elseif oh < hs && ow >= we
            src = b * pitch + hs * w + (we - 1) + 1
        elseif oh >= he && ow < ws
            src = b * pitch + (he - 1) * w + ws + 1
        elseif oh >= he && ow >= we
            src = b * pitch + (he - 1) * w + (we - 1) + 1
        end
        output[loc] = output[src]
    end
end

function run_zoom(repeat, input_sizes, zoom_factor)
    n, c, h, w = input_sizes
    ho = floor(Int, h * zoom_factor[1])
    wo = floor(Int, w * zoom_factor[2])
    is_zoom_out = ho < h && wo < w
    is_zoom_in = ho > h && wo > w
    if !is_zoom_out && !is_zoom_in
        println("Zoom factors only handle simultaneous expansion(or shrinkage) in both dimensions. Exit")
        return 1
    end

    pitch = h * w
    pad_h0 = pad_h1 = pad_w0 = pad_w1 = 0
    slice_h0 = slice_h1 = slice_w0 = slice_w1 = 0
    diff = h - ho
    half = abs(diff) ÷ 2
    if diff > 0
        pad_h0 = half
        pad_h1 = diff - half
    else
        slice_h0 = half
        slice_h1 = h + half
    end
    diff = w - wo
    half = abs(diff) ÷ 2
    if diff > 0
        pad_w0 = half
        pad_w1 = diff - half
    else
        slice_w0 = half
        slice_w1 = w + half
    end

    img_size = pitch * n * c
    rng = MersenneTwister(123)
    input = randn(rng, Float32, img_size)
    d_input = CuArray(input)
    d_output = CUDA.zeros(Float32, img_size)
    output_ref = zeros(Float32, img_size)
    block = (16, 16)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        CUDA.fill!(d_output, 0.0f0)
        if is_zoom_in
            grid = (cld(wo, 16), cld(ho, 16), n * c)
            @cuda threads=block blocks=grid zoom_in_kernel!(
                d_input, d_output, Int32(h), Int32(w), Int32(ho), Int32(wo),
                Int32(pitch), Int32(slice_h0), Int32(slice_h1), Int32(slice_w0), Int32(slice_w1))
        else
            grid = (cld(wo, 16), cld(ho, 16), n * c)
            @cuda threads=block blocks=grid zoom_out_kernel!(
                d_input, d_output, Int32(h), Int32(w), Int32(ho), Int32(wo),
                Int32(pitch), Int32(pad_h0), Int32(pad_h1), Int32(pad_w0), Int32(pad_w1))
            grid2 = (cld(w, 16), cld(h, 16), n * c)
            @cuda threads=block blocks=grid2 zoom_out_edge_pad_kernel!(
                d_output, Int32(h), Int32(w), Int32(pitch), Int32(pad_h0), Int32(pad_w0),
                Int32(pad_h0 + ho), Int32(pad_w0 + wo))
        end
    end
    CUDA.synchronize()
    @printf("Average execution time of the %s kernel: %f (us)\n",
            is_zoom_in ? "zoom-in" : "zoom-out", (time_ns() - t0) * 1.0e-3 / repeat)

    output = Array(d_output)
    if is_zoom_in
        zoom_in_ref!(input, output_ref, h, w, ho, wo, pitch, slice_h0, slice_h1, slice_w0, slice_w1, n * c)
    else
        zoom_out_ref!(input, output_ref, h, w, ho, wo, pitch, pad_h0, pad_h1, pad_w0, pad_w1, n * c)
        edge_pad_ref!(output_ref, h, w, pitch, pad_h0, pad_w0, pad_h0 + ho, pad_w0 + wo, n * c)
    end
    ok = all(abs.(output .- output_ref) .<= 1.0f-4)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

function main(args)
    if length(args) != 5
        println("Usage: main.jl <batch> <channel> <height> <width> <repeat>")
        return 1
    end
    input_sizes = (parse(Int, args[1]), parse(Int, args[2]), parse(Int, args[3]), parse(Int, args[4]))
    repeat = parse(Int, args[5])
    rc = run_zoom(repeat, input_sizes, (1.5f0, 2.5f0))
    rc |= run_zoom(repeat, input_sizes, (0.6f0, 0.9f0))
    return rc
end

exit(main(ARGS))
