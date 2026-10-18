using CUDA
using Printf

const MAX_POINTS_NUM = UInt32(300000)
const THREADS_FOR_VOXEL = 256

const MIN_X_RANGE = -54.0f0
const MAX_X_RANGE = 54.0f0
const MIN_Y_RANGE = -54.0f0
const MAX_Y_RANGE = 54.0f0
const MIN_Z_RANGE = -5.0f0
const MAX_Z_RANGE = 3.0f0
const PILLAR_X_SIZE = 0.075f0
const PILLAR_Y_SIZE = 0.075f0
const PILLAR_Z_SIZE = 0.2f0
const MAX_POINTS_PER_VOXEL = Int32(10)
const MAX_VOXELS = UInt32(160000)
const FEATURE_NUM = Int32(5)
const GRID_X_SIZE = Int32(round((MAX_X_RANGE - MIN_X_RANGE) / PILLAR_X_SIZE))
const GRID_Y_SIZE = Int32(round((MAX_Y_RANGE - MIN_Y_RANGE) / PILLAR_Y_SIZE))
const GRID_Z_SIZE = Int32(round((MAX_Z_RANGE - MIN_Z_RANGE) / PILLAR_Z_SIZE))

function device_hash(k::UInt64)
    k ⊻= k >> 16
    k *= UInt64(0x85ebca6b)
    k ⊻= k >> 13
    k *= UInt64(0xc2b2ae35)
    k ⊻= k >> 16
    return k
end

function insert_hash_table!(key::UInt32, value, hash_size::UInt32, hash_table)
    half = hash_size >>> 1
    slot = UInt32(device_hash(UInt64(key)) % UInt64(half))
    empty_key = typemax(UInt32)
    while true
        idx = Int(slot) + 1
        pre_key = CUDA.atomic_cas!(pointer(hash_table, idx), empty_key, key)
        if pre_key == empty_key
            hash_table[Int(slot + half) + 1] = CUDA.atomic_add!(pointer(value, 1), UInt32(1))
            return
        elseif pre_key == key
            return
        end
        slot = (slot + UInt32(1)) % half
    end
    return
end

function lookup_hash_table(key::UInt32, hash_size::UInt32, hash_table)
    slot = UInt32(device_hash(UInt64(key)) % UInt64(hash_size))
    empty_key = typemax(UInt32)
    while true
        idx = Int(slot) + 1
        found = hash_table[idx]
        if found == key
            return hash_table[Int(slot + hash_size) + 1]
        elseif found == empty_key
            return empty_key
        end
        slot = (slot + UInt32(1)) % hash_size
    end
    return empty_key
end

function build_hash_kernel!(points, points_size::Int32, hash_table, real_voxel_num)
    point_idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if point_idx >= points_size
        return
    end

    base = Int(point_idx * FEATURE_NUM) + 1
    px = points[base]
    py = points[base + 1]
    pz = points[base + 2]

    voxel_idx = Int32(floor((px - MIN_X_RANGE) / PILLAR_X_SIZE))
    if voxel_idx < 0 || voxel_idx >= GRID_X_SIZE
        return
    end
    voxel_idy = Int32(floor((py - MIN_Y_RANGE) / PILLAR_Y_SIZE))
    if voxel_idy < 0 || voxel_idy >= GRID_Y_SIZE
        return
    end
    voxel_idz = Int32(floor((pz - MIN_Z_RANGE) / PILLAR_Z_SIZE))
    if voxel_idz < 0 || voxel_idz >= GRID_Z_SIZE
        return
    end

    voxel_offset = UInt32(voxel_idz * GRID_Y_SIZE * GRID_X_SIZE +
                          voxel_idy * GRID_X_SIZE + voxel_idx)
    insert_hash_table!(voxel_offset, real_voxel_num, UInt32(points_size) * UInt32(4), hash_table)
    return
end

function voxelization_kernel!(points, points_size::Int32, hash_table, num_points_per_voxel,
                              voxels_temp, voxel_indices)
    point_idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if point_idx >= points_size
        return
    end

    base = Int(point_idx * FEATURE_NUM) + 1
    px = points[base]
    py = points[base + 1]
    pz = points[base + 2]

    if px < MIN_X_RANGE || px >= MAX_X_RANGE ||
       py < MIN_Y_RANGE || py >= MAX_Y_RANGE ||
       pz < MIN_Z_RANGE || pz >= MAX_Z_RANGE
        return
    end

    voxel_idx = Int32(floor((px - MIN_X_RANGE) / PILLAR_X_SIZE))
    voxel_idx >= GRID_X_SIZE && return
    voxel_idy = Int32(floor((py - MIN_Y_RANGE) / PILLAR_Y_SIZE))
    voxel_idy >= GRID_Y_SIZE && return
    voxel_idz = Int32(floor((pz - MIN_Z_RANGE) / PILLAR_Z_SIZE))
    voxel_idz >= GRID_Z_SIZE && return

    voxel_offset = UInt32(voxel_idz * GRID_Y_SIZE * GRID_X_SIZE +
                          voxel_idy * GRID_X_SIZE + voxel_idx)
    voxel_id = lookup_hash_table(voxel_offset, UInt32(points_size) * UInt32(2), hash_table)
    if voxel_id >= MAX_VOXELS
        return
    end

    vpos = Int(voxel_id) + 1
    current_num = CUDA.atomic_add!(pointer(num_points_per_voxel, vpos), UInt32(1))
    if current_num < UInt32(MAX_POINTS_PER_VOXEL)
        dst_offset = Int(voxel_id) * Int(FEATURE_NUM * MAX_POINTS_PER_VOXEL) +
                     Int(current_num) * Int(FEATURE_NUM)
        src_offset = Int(point_idx * FEATURE_NUM)
        for feature_idx in Int32(0):Int32(FEATURE_NUM - 1)
            voxels_temp[dst_offset + Int(feature_idx) + 1] =
                points[src_offset + Int(feature_idx) + 1]
        end
        idx_offset = Int(voxel_id) * 4
        voxel_indices[idx_offset + 1] = UInt32(0)
        voxel_indices[idx_offset + 2] = UInt32(voxel_idx)
        voxel_indices[idx_offset + 3] = UInt32(voxel_idy)
        voxel_indices[idx_offset + 4] = UInt32(voxel_idz)
    end
    return
end

function feature_extraction_kernel!(voxels_temp, num_points_per_voxel, real_voxel_num::UInt32,
                                    voxel_features)
    voxel_idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if UInt32(voxel_idx0) >= real_voxel_num
        return
    end

    voxel_idx = Int(voxel_idx0)
    valid_points_num = num_points_per_voxel[voxel_idx + 1]
    if valid_points_num > UInt32(MAX_POINTS_PER_VOXEL)
        num_points_per_voxel[voxel_idx + 1] = UInt32(MAX_POINTS_PER_VOXEL)
        valid_points_num = UInt32(MAX_POINTS_PER_VOXEL)
    end

    offset = voxel_idx * Int(MAX_POINTS_PER_VOXEL * FEATURE_NUM)
    for feature_idx in Int32(0):Int32(FEATURE_NUM - 1)
        s = voxels_temp[offset + Int(feature_idx) + 1]
        for point_idx in Int32(0):(Int32(valid_points_num) - Int32(2))
            s += voxels_temp[offset + Int((point_idx + Int32(1)) * FEATURE_NUM + feature_idx) + 1]
        end
        voxels_temp[offset + Int(feature_idx) + 1] = s / Float32(valid_points_num)
    end

    for feature_idx in Int32(0):Int32(FEATURE_NUM - 1)
        dst_offset = voxel_idx * Int(FEATURE_NUM)
        src_offset = voxel_idx * Int(FEATURE_NUM * MAX_POINTS_PER_VOXEL)
        voxel_features[dst_offset + Int(feature_idx) + 1] =
            Float16(voxels_temp[src_offset + Int(feature_idx) + 1])
    end
    return
end

function generate_voxels(points::CuArray{Float32}, points_size::Int, repeat::Int)
    hash_table = CUDA.fill(typemax(UInt32), Int(MAX_POINTS_NUM) * 4)
    voxels_temp = CUDA.fill(reinterpret(Float32, 0xffffffff), Int(MAX_VOXELS) *
                            Int(MAX_POINTS_PER_VOXEL) * Int(FEATURE_NUM))
    voxel_features = CUDA.zeros(Float16, Int(MAX_VOXELS) * Int(MAX_POINTS_PER_VOXEL) * Int(FEATURE_NUM))
    voxel_num = CUDA.zeros(UInt32, Int(MAX_VOXELS))
    voxel_indices = CUDA.zeros(UInt32, Int(MAX_VOXELS) * 4)
    real_num_voxels = CUDA.zeros(UInt32, 1)

    CUDA.synchronize()
    blocks = cld(points_size, THREADS_FOR_VOXEL)
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS_FOR_VOXEL blocks=blocks build_hash_kernel!(
            points, Int32(points_size), hash_table, real_num_voxels)
        @cuda threads=THREADS_FOR_VOXEL blocks=blocks voxelization_kernel!(
            points, Int32(points_size), hash_table, voxel_num, voxels_temp, voxel_indices)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average execution time of the voxelization kernel: %f (us)\n",
            elapsed * 1.0e-3 / repeat)

    real_count = CUDA.@allowscalar real_num_voxels[1]
    println("valid_num: ", real_count)

    feature_blocks = cld(Int(real_count), THREADS_FOR_VOXEL)
    start = time_ns()
    for _ in 1:repeat
        if feature_blocks > 0
            @cuda threads=THREADS_FOR_VOXEL blocks=feature_blocks feature_extraction_kernel!(
                voxels_temp, voxel_num, real_count, voxel_features)
        end
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average execution time of the feature extraction kernel: %f (us)\n",
            elapsed * 1.0e-3 / repeat)
end

function load_points(path::String)
    bytes = read(path)
    vals = reinterpret(Float32, bytes)
    return collect(vals)
end

function main()
    if length(ARGS) < 2
        println("Usage: ")
        println("    ./main data/test/  100")
        println("    Run voxelization with data under ./data/test/ for 100 times")
        exit(0)
    end

    display_folder = ARGS[1]
    data_folder = display_folder
    if !isdir(data_folder) && isdir(joinpath("..", "voxelization-cuda", data_folder))
        data_folder = joinpath("..", "voxelization-cuda", data_folder)
    end
    repeat = parse(Int, ARGS[2])
    files = filter(f -> endswith(f, ".bin"), readdir(data_folder; sort=false))
    println("Number of files: ", length(files))

    for file in files
        data_file = joinpath(data_folder, file)
        println()
        println("<<<<<<<<<<<")
        println("load file: ", joinpath(display_folder, file))
        points = load_points(data_file)
        points_num = length(points) ÷ Int(FEATURE_NUM)
        println("find points num: ", points_num)
        d_points = CuArray(points)
        generate_voxels(d_points, points_num, repeat)
        println(">>>>>>>>>>>")
    end
end

main()
