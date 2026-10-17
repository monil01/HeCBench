function multinomial_row!(cur_dist,dest,distributions,categories,sampled,dist,stride_dist,stride_categories)
    stride_grid = Int32(512)
    while cur_dist < distributions
        if true
            sum = 0.0f0
            for cat in Int32(0):(categories - Int32(1))
                @inbounds sum += dist[cur_dist * stride_dist + cat * stride_categories + Int32(1)]
            end
            if sum == 0.0f0
                @inbounds dest[cur_dist + Int32(1)] = Int32(0)
            else
                sample = @inbounds sampled[cur_dist + Int32(1)]
                prev_bucket = 0.0f0
                found = false
                found_pos = Int32(0)
                for cat in Int32(0):(categories - Int32(1))
                    dist_val = @inbounds dist[cur_dist * stride_dist + cat * stride_categories + Int32(1)] / sum
                    cur_bucket = prev_bucket + dist_val
                    if sample < cur_bucket && sample >= prev_bucket && dist_val > 0.0f0
                        found_pos = cat
                        found = true
                        break
                    end
                    prev_bucket = cur_bucket
                end
                if found
                    @inbounds dest[cur_dist + Int32(1)] = found_pos
                else
                    for cat in (categories - Int32(1)):-Int32(1):Int32(0)
                        if @inbounds dist[cur_dist * stride_dist + cat * stride_categories + Int32(1)] > 0.0f0
                            @inbounds dest[cur_dist + Int32(1)] = cat
                            break
                        end
                    end
                end
            end
        end
        cur_dist += stride_grid
    end
    return
end
