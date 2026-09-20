using CUDA
using Printf
using Random

const NUM_CLASSES = 3
const MIN_X_RANGE = Float32(0.0)
const MAX_X_RANGE = Float32(69.12)
const MIN_Y_RANGE = Float32(-39.68)
const MAX_Y_RANGE = Float32(39.68)
const DIR_OFFSET = Float32(0.78539)
const NUM_DIR_BINS = 2
const NUM_ANCHORS = NUM_CLASSES * 2
const LEN_PER_ANCHOR = 4
const ANCHORS = Float32[
    3.9, 1.6, 1.56, 0.0,
    3.9, 1.6, 1.56, 1.57,
    0.8, 0.6, 1.73, 0.0,
    0.8, 0.6, 1.73, 1.57,
    1.76, 0.6, 1.73, 0.0,
    1.76, 0.6, 1.73, 1.57,
]
const ANCHOR_BOTTOM_HEIGHTS = Float32[-1.78, -0.6, -0.6]
const SCORE_THRESH = Float32(0.1)
const NUM_BOX_VALUES = 7
const GRID_X_SIZE = 432
const GRID_Y_SIZE = 496
const FEATURE_X_SIZE = GRID_X_SIZE ÷ 2
const FEATURE_Y_SIZE = GRID_Y_SIZE ÷ 2

function postprocess_kernel!(cls_input, box_input, dir_cls_input, anchors,
                             anchor_bottom_heights, bndbox_output, score_output,
                             object_counter, min_x_range::Float32, max_x_range::Float32,
                             min_y_range::Float32, max_y_range::Float32,
                             feature_x_size::Int32, feature_y_size::Int32,
                             num_anchors::Int32, num_classes::Int32,
                             num_box_values::Int32, score_thresh::Float32,
                             dir_offset::Float32)
    loc_index = blockIdx().x - Int32(1)
    ith_anchor = threadIdx().x - Int32(1)
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

        res_count = CUDA.atomic_add!(pointer(object_counter), Int32(1))
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

function reference!(cls_input, box_input, dir_cls_input, anchors, anchor_bottom_heights,
                    bndbox_output, score_output)
    res_count = 0
    feature_size = FEATURE_X_SIZE * FEATURE_Y_SIZE
    for loc_index in 0:(feature_size - 1)
        col = loc_index % FEATURE_X_SIZE
        row = loc_index ÷ FEATURE_X_SIZE
        x_offset = MIN_X_RANGE + Float32(col) * (MAX_X_RANGE - MIN_X_RANGE) / Float32(FEATURE_X_SIZE - 1)
        y_offset = MIN_Y_RANGE + Float32(row) * (MAX_Y_RANGE - MIN_Y_RANGE) / Float32(FEATURE_Y_SIZE - 1)
        for ith_anchor in 0:(NUM_ANCHORS - 1)
            cls_offset = loc_index * NUM_ANCHORS * NUM_CLASSES + ith_anchor * NUM_CLASSES
            max_score = Float32(1) / (Float32(1) + exp(-cls_input[cls_offset + 1]))
            cls_id = 0
            for i in 1:(NUM_CLASSES - 1)
                cls_score = Float32(1) / (Float32(1) + exp(-cls_input[cls_offset + i + 1]))
                if cls_score > max_score
                    max_score = cls_score
                    cls_id = i
                end
            end

            if max_score >= SCORE_THRESH
                box_offset = loc_index * NUM_ANCHORS * NUM_BOX_VALUES + ith_anchor * NUM_BOX_VALUES
                dir_cls_offset = loc_index * NUM_ANCHORS * 2 + ith_anchor * 2
                anchor_offset = ith_anchor * 4
                dxa = anchors[anchor_offset + 1]
                dya = anchors[anchor_offset + 2]
                dza = anchors[anchor_offset + 3]
                ra = anchors[anchor_offset + 4]
                za = dza / Float32(2) + anchor_bottom_heights[(ith_anchor ÷ 2) + 1]
                diagonal = sqrt(dxa * dxa + dya * dya)

                box_input[box_offset + 1] = box_input[box_offset + 1] * diagonal + x_offset
                box_input[box_offset + 2] = box_input[box_offset + 2] * diagonal + y_offset
                box_input[box_offset + 3] = box_input[box_offset + 3] * dza + za
                box_input[box_offset + 4] = exp(box_input[box_offset + 4]) * dxa
                box_input[box_offset + 5] = exp(box_input[box_offset + 5]) * dya
                box_input[box_offset + 6] = exp(box_input[box_offset + 6]) * dza
                box_input[box_offset + 7] = box_input[box_offset + 7] + ra

                dir_label = dir_cls_input[dir_cls_offset + 1] > dir_cls_input[dir_cls_offset + 2] ? 0 : 1
                period = Float32(pi)
                val = box_input[box_offset + 7] - DIR_OFFSET
                dir_rot = val - floor(val / (period + Float32(1.0f-8))) * period
                yaw = dir_rot + DIR_OFFSET + period * Float32(dir_label)

                base = res_count * 9
                bndbox_output[base + 1] = box_input[box_offset + 1]
                bndbox_output[base + 2] = box_input[box_offset + 2]
                bndbox_output[base + 3] = box_input[box_offset + 3]
                bndbox_output[base + 4] = box_input[box_offset + 4]
                bndbox_output[base + 5] = box_input[box_offset + 5]
                bndbox_output[base + 6] = box_input[box_offset + 6]
                bndbox_output[base + 7] = yaw
                bndbox_output[base + 8] = Float32(cls_id)
                bndbox_output[base + 9] = Float32(box_offset)
                score_output[res_count + 1] = max_score
                res_count += 1
            end
        end
    end
    return res_count
end

function verify_outputs(bndbox_num, cls_input, box_input, dir_cls_input, bndbox_output, score_output)
    feature_anchor_size = FEATURE_X_SIZE * FEATURE_Y_SIZE * NUM_ANCHORS
    score_ref = Vector{Float32}(undef, feature_anchor_size)
    box_ref = Vector{Float32}(undef, feature_anchor_size * 9)
    box_input_ref = copy(box_input)
    bndbox_num_ref = reference!(cls_input, box_input_ref, dir_cls_input, ANCHORS,
                                ANCHOR_BOTTOM_HEIGHTS, box_ref, score_ref)

    ok = bndbox_num_ref == bndbox_num
    if ok && bndbox_num > 0
        max_val, loc = findmax(@view score_output[1:bndbox_num])
        found = false
        for i in 1:bndbox_num_ref
            if abs(max_val - score_ref[i]) < Float32(1.0f-6)
                box_base = (loc - 1) * 9
                ref_base = (i - 1) * 9
                if bndbox_output[box_base + 9] == box_ref[ref_base + 9]
                    @printf("Comparing values at location %d and reference location %d\n", loc - 1, i - 1)
                    found = true
                    for j in 1:9
                        if abs(bndbox_output[box_base + j] - box_ref[ref_base + j]) > Float32(1.0f-3)
                            ok = false
                            break
                        end
                    end
                    break
                end
            end
        end
        ok &= found
    end
    println(ok ? "PASS" : "FAIL")
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])

    feature_size = FEATURE_X_SIZE * FEATURE_Y_SIZE
    feature_anchor_size = feature_size * NUM_ANCHORS
    cls_size = feature_anchor_size * NUM_CLASSES
    box_size = feature_anchor_size * NUM_BOX_VALUES
    dir_cls_size = feature_anchor_size * NUM_DIR_BINS
    bndbox_size = feature_anchor_size * 9

    rng = MersenneTwister(123)
    h_cls_input = rand(rng, Float32, cls_size)
    h_box_input = rand(rng, Float32, box_size)
    h_dir_cls_input = rand(rng, Float32, dir_cls_size)

    d_cls_input = CuArray(h_cls_input)
    d_box_input = CuArray(h_box_input)
    d_dir_cls_input = CuArray(h_dir_cls_input)
    d_anchors = CuArray(ANCHORS)
    d_anchor_bottom_heights = CuArray(ANCHOR_BOTTOM_HEIGHTS)
    d_bndbox_output = CUDA.zeros(Float32, bndbox_size)
    d_score_output = CUDA.zeros(Float32, feature_anchor_size)
    d_object_counter = CUDA.zeros(Int32, 1)

    time_ns_total = Int128(0)
    threads = NUM_ANCHORS
    blocks = feature_size
    for _ in 1:repeat
        copyto!(d_box_input, h_box_input)
        CUDA.fill!(d_object_counter, Int32(0))
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=threads blocks=blocks postprocess_kernel!(
            d_cls_input, d_box_input, d_dir_cls_input, d_anchors,
            d_anchor_bottom_heights, d_bndbox_output, d_score_output,
            d_object_counter, MIN_X_RANGE, MAX_X_RANGE, MIN_Y_RANGE, MAX_Y_RANGE,
            Int32(FEATURE_X_SIZE), Int32(FEATURE_Y_SIZE), Int32(NUM_ANCHORS),
            Int32(NUM_CLASSES), Int32(NUM_BOX_VALUES), SCORE_THRESH, DIR_OFFSET)
        CUDA.synchronize()
        time_ns_total += Int128(time_ns() - start)
    end

    @printf("Average execution time of postprocess kernel: %f (us)\n",
            Float64(time_ns_total) * 1e-3 / repeat)

    bndbox_num = Int(Array(d_object_counter)[1])
    h_bndbox_output = Array(d_bndbox_output)
    h_score_output = Array(d_score_output)
    verify_outputs(bndbox_num, h_cls_input, h_box_input, h_dir_cls_input,
                   h_bndbox_output, h_score_output)
    return 0
end

exit(main(ARGS))
