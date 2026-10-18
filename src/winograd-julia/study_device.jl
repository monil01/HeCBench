using StaticArrays
const MAP_SIZE=131
const OUT_SIZE=129
const TILE_N=65
@inline function winograd_tile(input, transformed_filter, tile_i::Int, tile_j::Int)
    input_tile = MMatrix{4, 4, Float32}(undef)
    tmp_tile = MMatrix{4, 4, Float32}(undef)
    transformed_tile = MMatrix{4, 4, Float32}(undef)
    multiplied_tile = MMatrix{4, 4, Float32}(undef)
    tmp_tile_1 = MMatrix{2, 4, Float32}(undef)
    final_tile = MMatrix{2, 2, Float32}(undef)

    @inbounds for i in 0:3
        for j in 0:3
            x = 2 * tile_i + i
            y = 2 * tile_j + j
            input_tile[i + 1, j + 1] = (x >= MAP_SIZE || y >= MAP_SIZE) ? Float32(0) : input[x * MAP_SIZE + y + 1]
        end
    end

    @inbounds for j in 1:4
        tmp_tile[1, j] = input_tile[1, j] - input_tile[3, j]
        tmp_tile[2, j] = input_tile[2, j] + input_tile[3, j]
        tmp_tile[3, j] = -input_tile[2, j] + input_tile[3, j]
        tmp_tile[4, j] = input_tile[2, j] - input_tile[4, j]
    end
    @inbounds for i in 1:4
        transformed_tile[i, 1] = tmp_tile[i, 1] - tmp_tile[i, 3]
        transformed_tile[i, 2] = tmp_tile[i, 2] + tmp_tile[i, 3]
        transformed_tile[i, 3] = -tmp_tile[i, 2] + tmp_tile[i, 3]
        transformed_tile[i, 4] = tmp_tile[i, 2] - tmp_tile[i, 4]
    end

    @inbounds for i in 1:4
        for j in 1:4
            multiplied_tile[i, j] = transformed_tile[i, j] * transformed_filter[(i - 1) * 4 + j]
        end
    end

    @inbounds for j in 1:4
        tmp_tile_1[1, j] = multiplied_tile[1, j] + multiplied_tile[2, j] + multiplied_tile[3, j]
        tmp_tile_1[2, j] = multiplied_tile[2, j] - multiplied_tile[3, j] - multiplied_tile[4, j]
    end
    @inbounds for i in 1:2
        final_tile[i, 1] = tmp_tile_1[i, 1] + tmp_tile_1[i, 2] + tmp_tile_1[i, 3]
        final_tile[i, 2] = tmp_tile_1[i, 2] - tmp_tile_1[i, 3] - tmp_tile_1[i, 4]
    end
    return final_tile
end


@inline function store_tile!(input,filter,output,ti,tj)
 result=winograd_tile(input,filter,Int(ti),Int(tj))
 for i in 0:1,j in 0:1
  x=2ti+i;y=2tj+j
  if x<OUT_SIZE && y<OUT_SIZE
   @inbounds output[x*OUT_SIZE+y+1]=result[i+1,j+1]
  end
 end
 return
end
function cpu_tiles!(input,filter,output,rows)
 for ti in 0:rows-1,tj in 0:TILE_N-1
  store_tile!(input,filter,output,ti,tj)
 end
 return
end
