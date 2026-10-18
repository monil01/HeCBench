# Pure worker extracted from the byte-preserved existing CUDA.jl reference.
function dwconv_lane!(linear::Int32,input, output, weight, bias,
                        total::Int32, output_channels::Int32, depthwise_multiplier::Int32,
                        input_width::Int32, input_height::Int32,
                        output_width::Int32, output_height::Int32,
                        kernel_width::Int32, kernel_height::Int32,
                        stride_width::Int32, stride_height::Int32,
                        pad_width::Int32, pad_height::Int32,
                        dilation_width::Int32, dilation_height::Int32)
    if linear >= total
        return
    end

    tmp1 = linear ÷ output_width
    w = linear - tmp1 * output_width
    tmp2 = tmp1 ÷ output_height
    h = tmp1 - tmp2 * output_height
    tmp1 = tmp2
    tmp2 = tmp1 ÷ output_channels
    c = tmp1 - tmp2 * output_channels
    n = tmp2

    input_channel = c
    input_channels = output_channels
    if depthwise_multiplier != Int32(1)
        input_channel = input_channel ÷ depthwise_multiplier
        input_channels = input_channels ÷ depthwise_multiplier
    end

    weight_offset = c * kernel_height * kernel_width
    value = @inbounds bias[c + Int32(1)]
    offset0 = (n * input_channels + input_channel) * input_height * input_width
    k_h = Int32(0)
    while k_h < kernel_height
        k_w = Int32(0)
        while k_w < kernel_width
            h_in = -pad_height + h * stride_height + k_h * dilation_height
            w_in = -pad_width + w * stride_width + k_w * dilation_width
            if h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width
                offset = offset0 + h_in * input_width + w_in
                @inbounds value += weight[weight_offset + Int32(1)] * input[offset + Int32(1)]
            end
            weight_offset += Int32(1)
            k_w += Int32(1)
        end
        k_h += Int32(1)
    end
    @inbounds output[linear + Int32(1)] = value
    return
end

