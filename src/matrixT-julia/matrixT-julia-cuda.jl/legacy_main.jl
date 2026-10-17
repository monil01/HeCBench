using CUDA
using Printf

const TILE_DIM = Int32(16)
const BLOCK_ROWS = Int32(16)

@inline function idx2(x::Int32, y::Int32, width::Int32)
    return Int(x + width * y + Int32(1))
end

function copy_kernel!(odata, idata, width::Int32, height::Int32)
    x = (blockIdx().x - Int32(1)) * TILE_DIM + threadIdx().x - Int32(1)
    y = (blockIdx().y - Int32(1)) * TILE_DIM + threadIdx().y - Int32(1)
    index = x + width * y
    @inbounds odata[Int(index + Int32(1))] = idata[Int(index + Int32(1))]
    return
end

function copy_shared_kernel!(odata, idata, width::Int32, height::Int32)
    tile = @cuStaticSharedMem(Float32, (16, 16))
    tx = threadIdx().x
    ty = threadIdx().y
    x = (blockIdx().x - Int32(1)) * TILE_DIM + tx - Int32(1)
    y = (blockIdx().y - Int32(1)) * TILE_DIM + ty - Int32(1)
    if x < width && y < height
        @inbounds tile[Int(ty), Int(tx)] = idata[idx2(x, y, width)]
    end
    sync_threads()
    if x < height && y < width
        @inbounds odata[idx2(x, y, width)] = tile[Int(ty), Int(tx)]
    end
    return
end

function transpose_naive_kernel!(odata, idata, width::Int32, height::Int32)
    x = (blockIdx().x - Int32(1)) * TILE_DIM + threadIdx().x - Int32(1)
    y = (blockIdx().y - Int32(1)) * TILE_DIM + threadIdx().y - Int32(1)
    @inbounds odata[Int(y + height * x + Int32(1))] = idata[idx2(x, y, width)]
    return
end

function transpose_coalesced_kernel!(odata, idata, width::Int32, height::Int32)
    tile = @cuStaticSharedMem(Float32, (16, 16))
    tx = threadIdx().x
    ty = threadIdx().y
    x_in = (blockIdx().x - Int32(1)) * TILE_DIM + tx - Int32(1)
    y_in = (blockIdx().y - Int32(1)) * TILE_DIM + ty - Int32(1)
    @inbounds tile[Int(ty), Int(tx)] = idata[idx2(x_in, y_in, width)]
    sync_threads()
    x_out = (blockIdx().y - Int32(1)) * TILE_DIM + tx - Int32(1)
    y_out = (blockIdx().x - Int32(1)) * TILE_DIM + ty - Int32(1)
    @inbounds odata[idx2(x_out, y_out, height)] = tile[Int(tx), Int(ty)]
    return
end

function transpose_pad_kernel!(odata, idata, width::Int32, height::Int32)
    tile = @cuStaticSharedMem(Float32, (17, 16))
    tx = threadIdx().x
    ty = threadIdx().y
    x_in = (blockIdx().x - Int32(1)) * TILE_DIM + tx - Int32(1)
    y_in = (blockIdx().y - Int32(1)) * TILE_DIM + ty - Int32(1)
    @inbounds tile[Int(ty), Int(tx)] = idata[idx2(x_in, y_in, width)]
    sync_threads()
    x_out = (blockIdx().y - Int32(1)) * TILE_DIM + tx - Int32(1)
    y_out = (blockIdx().x - Int32(1)) * TILE_DIM + ty - Int32(1)
    @inbounds odata[idx2(x_out, y_out, height)] = tile[Int(tx), Int(ty)]
    return
end

@inline swizzle(row::Int32, col::Int32) = col ⊻ row

function transpose_swizzle_kernel!(odata, idata, width::Int32, height::Int32)
    tile = @cuStaticSharedMem(Float32, (16, 16))
    tx = threadIdx().x
    ty = threadIdx().y
    x_in = (blockIdx().x - Int32(1)) * TILE_DIM + tx - Int32(1)
    y_in = (blockIdx().y - Int32(1)) * TILE_DIM + ty - Int32(1)
    col = swizzle(ty - Int32(1), tx - Int32(1)) + Int32(1)
    @inbounds tile[Int(ty), Int(col)] = idata[idx2(x_in, y_in, width)]
    sync_threads()
    x_out = (blockIdx().y - Int32(1)) * TILE_DIM + tx - Int32(1)
    y_out = (blockIdx().x - Int32(1)) * TILE_DIM + ty - Int32(1)
    col2 = swizzle(tx - Int32(1), ty - Int32(1)) + Int32(1)
    @inbounds odata[idx2(x_out, y_out, height)] = tile[Int(tx), Int(col2)]
    return
end

function transpose_diagonal_kernel!(odata, idata, width::Int32, height::Int32)
    tile = @cuStaticSharedMem(Float32, (17, 16))
    bx = blockIdx().x - Int32(1)
    by = blockIdx().y - Int32(1)
    gx = gridDim().x
    gy = gridDim().y
    if width == height
        block_y = bx
        block_x = (bx + block_y) % gx
    else
        bid = bx + gx * by
        block_y = bid % gy
        block_x = ((bid ÷ gy) + block_y) % gx
    end
    tx = threadIdx().x
    ty = threadIdx().y
    x_in = block_x * TILE_DIM + tx - Int32(1)
    y_in = block_y * TILE_DIM + ty - Int32(1)
    @inbounds tile[Int(ty), Int(tx)] = idata[idx2(x_in, y_in, width)]
    sync_threads()
    x_out = block_y * TILE_DIM + tx - Int32(1)
    y_out = block_x * TILE_DIM + ty - Int32(1)
    @inbounds odata[idx2(x_out, y_out, height)] = tile[Int(tx), Int(ty)]
    return
end

function transpose_fine_kernel!(odata, idata, width::Int32, height::Int32)
    block = @cuStaticSharedMem(Float32, (17, 16))
    tx = threadIdx().x
    ty = threadIdx().y
    x = (blockIdx().x - Int32(1)) * TILE_DIM + tx - Int32(1)
    y = (blockIdx().y - Int32(1)) * TILE_DIM + ty - Int32(1)
    @inbounds block[Int(ty), Int(tx)] = idata[idx2(x, y, width)]
    sync_threads()
    @inbounds odata[idx2(x, y, height)] = block[Int(tx), Int(ty)]
    return
end

function transpose_coarse_kernel!(odata, idata, width::Int32, height::Int32)
    block = @cuStaticSharedMem(Float32, (17, 16))
    tx = threadIdx().x
    ty = threadIdx().y
    x_in = (blockIdx().x - Int32(1)) * TILE_DIM + tx - Int32(1)
    y_in = (blockIdx().y - Int32(1)) * TILE_DIM + ty - Int32(1)
    @inbounds block[Int(ty), Int(tx)] = idata[idx2(x_in, y_in, width)]
    sync_threads()
    x_out = (blockIdx().y - Int32(1)) * TILE_DIM + tx - Int32(1)
    y_out = (blockIdx().x - Int32(1)) * TILE_DIM + ty - Int32(1)
    @inbounds odata[idx2(x_out, y_out, height)] = block[Int(ty), Int(tx)]
    return
end

function compute_transpose_gold(idata, size_x::Int, size_y::Int)
    gold = Vector{Float32}(undef, size_x * size_y)
    @inbounds for y in 0:(size_y - 1)
        for x in 0:(size_x - 1)
            gold[x * size_y + y + 1] = idata[y * size_x + x + 1]
        end
    end
    return gold
end

function run_kernel!(name, kernel!, d_odata, d_idata, width::Int32, height::Int32, grid, threads, repeat::Int)
    @cuda threads=threads blocks=grid kernel!(d_odata, d_idata, width, height)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=grid kernel!(d_odata, d_idata, width, height)
    end
    CUDA.synchronize()
    @printf("Average kernel (%s) execution time: %f (us)\n", name, (time_ns() - t0) * 1e-3 / repeat)
end

function main()
    if length(ARGS) != 3
        println("\nCommand line options")
        println("\t<row_dim_size> (matrix row    dimensions)")
        println("\t<col_dim_size> (matrix column dimensions)")
        println("\t<repeat> (kernel execution count)")
        return
    end

    size_x = parse(Int, ARGS[1])
    size_y = parse(Int, ARGS[2])
    repeat = parse(Int, ARGS[3])

    if size_x != size_y
        @printf("Error: non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n", size_x, size_y)
        exit(1)
    end
    if size_x % Int(TILE_DIM) != 0 || size_y % Int(TILE_DIM) != 0
        println("Matrix size must be integral multiple of tile size\nExiting...\n")
        exit(1)
    end

    h_idata = Float32.(0:(size_x * size_y - 1))
    h_odata = similar(h_idata)
    transpose_gold = compute_transpose_gold(h_idata, size_x, size_y)
    d_idata = CuArray(h_idata)
    d_odata = CuArray(h_odata)

    @printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
            size_x, size_y, size_x ÷ Int(TILE_DIM), size_y ÷ Int(TILE_DIM),
            Int(TILE_DIM), Int(TILE_DIM), Int(TILE_DIM), Int(BLOCK_ROWS))

    grid = (size_x ÷ Int(TILE_DIM), size_y ÷ Int(TILE_DIM))
    threads = (Int(TILE_DIM), Int(BLOCK_ROWS))
    width = Int32(size_x)
    height = Int32(size_y)

    kernels = (
        ("simple memory copy  ", copy_kernel!, :copy),
        ("shared memory copy  ", copy_shared_kernel!, :copy),
        ("coarse-grained      ", transpose_coarse_kernel!, :bypass),
        ("fine-grained        ", transpose_fine_kernel!, :bypass),
        ("transpose naive     ", transpose_naive_kernel!, :transpose),
        ("transpose coalesced ", transpose_coalesced_kernel!, :transpose),
        ("transpose smem pad  ", transpose_pad_kernel!, :transpose),
        ("transpose swizzle   ", transpose_swizzle_kernel!, :transpose),
        ("transpose diagonal  ", transpose_diagonal_kernel!, :transpose),
    )

    success = true
    for (name, kernel!, mode) in kernels
        run_kernel!(name, kernel!, d_odata, d_idata, width, height, grid, threads, repeat)
        copyto!(h_odata, d_odata)
        gold = mode === :copy ? h_idata : mode === :bypass ? h_odata : transpose_gold
        ok = true
        @inbounds for i in eachindex(h_odata)
            if abs(gold[i] - h_odata[i]) > 0.01f0
                ok = false
                break
            end
        end
        if !ok
            println("*** $(name) kernel FAILED ***")
            success = false
        end
    end

    println(success ? "PASS" : "FAIL")
end

main()
