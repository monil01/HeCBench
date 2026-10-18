function zoom_in_ref!(input, output, h, w, ho, wo, pitch, hs, he, ws, we, batch)
    ratio_h = Float32(h) / Float32(ho)
    ratio_w = Float32(w) / Float32(wo)
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
    ratio_h = Float32(h) / Float32(ho)
    ratio_w = Float32(w) / Float32(wo)
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

