# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Genuine Julia math retained; portable Atomix reservation replaces CUDA atomic_add!.
import Atomix
@inline function portable_postprocess!(loc_index::Int32, ith_anchor::Int32, cls_input, box_input, dir_cls_input, anchors,
                             anchor_bottom_heights, bndbox_output, score_output,
                             object_counter, min_x_range::Float32, max_x_range::Float32,
                             min_y_range::Float32, max_y_range::Float32,
                             feature_x_size::Int32, feature_y_size::Int32,
                             num_anchors::Int32, num_classes::Int32,
                             num_box_values::Int32, score_thresh::Float32,
                             dir_offset::Float32)
    if ith_anchor >= num_anchors
        return
    end

    col = loc_index % feature_x_size
    row = loc_index ÷ feature_x_size
    x_offset = min_x_range + Float32(col) * (max_x_range - min_x_range) / Float32(feature_x_size - Int32(1))
    y_offset = min_y_range + Float32(row) * (max_y_range - min_y_range) / Float32(feature_y_size - Int32(1))
    cls_offset = loc_index * num_anchors * num_classes + ith_anchor * num_classes

    @inbounds max_score = Float32(1) / (Float32(1) + exp(-cls_input[Int(cls_offset) + 1]))
    cls_id = Int32(0)
    for i in Int32(1):(num_classes - Int32(1))
        @inbounds cls_score = Float32(1) / (Float32(1) + exp(-cls_input[Int(cls_offset + i) + 1]))
        if cls_score > max_score
            max_score = cls_score
            cls_id = i
        end
    end

    if max_score >= score_thresh
        box_offset = loc_index * num_anchors * num_box_values + ith_anchor * num_box_values
        dir_cls_offset = loc_index * num_anchors * Int32(2) + ith_anchor * Int32(2)
        anchor_offset = ith_anchor * Int32(4)
        @inbounds dxa = anchors[Int(anchor_offset) + 1]
        @inbounds dya = anchors[Int(anchor_offset) + 2]
        @inbounds dza = anchors[Int(anchor_offset) + 3]
        @inbounds ra = anchors[Int(anchor_offset) + 4]
        @inbounds za = dza / Float32(2) + anchor_bottom_heights[Int(ith_anchor ÷ Int32(2)) + 1]
        diagonal = sqrt(dxa * dxa + dya * dya)

        @inbounds box_input[Int(box_offset) + 1] = box_input[Int(box_offset) + 1] * diagonal + x_offset
        @inbounds box_input[Int(box_offset) + 2] = box_input[Int(box_offset) + 2] * diagonal + y_offset
        @inbounds box_input[Int(box_offset) + 3] = box_input[Int(box_offset) + 3] * dza + za
        @inbounds box_input[Int(box_offset) + 4] = exp(box_input[Int(box_offset) + 4]) * dxa
        @inbounds box_input[Int(box_offset) + 5] = exp(box_input[Int(box_offset) + 5]) * dya
        @inbounds box_input[Int(box_offset) + 6] = exp(box_input[Int(box_offset) + 6]) * dza
        @inbounds box_input[Int(box_offset) + 7] = box_input[Int(box_offset) + 7] + ra

        @inbounds dir_label = dir_cls_input[Int(dir_cls_offset) + 1] > dir_cls_input[Int(dir_cls_offset) + 2] ? Int32(0) : Int32(1)
        period = Float32(pi)
        @inbounds val = box_input[Int(box_offset) + 7] - dir_offset
        dir_rot = val - floor(val / (period + Float32(1.0f-8))) * period
        yaw = dir_rot + dir_offset + period * Float32(dir_label)

        res_count = (Atomix.@atomic :monotonic object_counter[1] += Int32(1)) - Int32(1)
        base = Int(res_count) * 9
        @inbounds bndbox_output[base + 1] = box_input[Int(box_offset) + 1]
        @inbounds bndbox_output[base + 2] = box_input[Int(box_offset) + 2]
        @inbounds bndbox_output[base + 3] = box_input[Int(box_offset) + 3]
        @inbounds bndbox_output[base + 4] = box_input[Int(box_offset) + 4]
        @inbounds bndbox_output[base + 5] = box_input[Int(box_offset) + 5]
        @inbounds bndbox_output[base + 6] = box_input[Int(box_offset) + 6]
        @inbounds bndbox_output[base + 7] = yaw
        @inbounds bndbox_output[base + 8] = Float32(cls_id)
        @inbounds bndbox_output[base + 9] = Float32(box_offset)
        @inbounds score_output[Int(res_count) + 1] = max_score
    end
    return
end
