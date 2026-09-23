using CUDA
using Printf
using Random

const K_HASH_TABLE_CAPACITY = UInt32(64 * 1024 * 1024)
const K_NUM_KEYVALUES = UInt32(K_HASH_TABLE_CAPACITY ÷ UInt32(2))
const K_EMPTY = typemax(UInt32)
const THREADS = 256

@inline function hash_key(k::UInt32)
    k ⊻= k >> UInt32(16)
    k *= UInt32(0x85ebca6b)
    k ⊻= k >> UInt32(13)
    k *= UInt32(0xc2b2ae35)
    k ⊻= k >> UInt32(16)
    return k & (K_HASH_TABLE_CAPACITY - UInt32(1))
end

function hashtable_insert_kernel!(table_keys, table_values, keys, values, num_kvs::UInt32)
    tid = UInt32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
    if tid <= num_kvs
        key = keys[tid]
        value = values[tid]
        slot = hash_key(key)

        while true
            # CUDA source is 0-based; array slots become 1-based only here.
            idx = slot + UInt32(1)
            prev = CUDA.atomic_cas!(pointer(table_keys, idx), K_EMPTY, key)
            if prev == K_EMPTY || prev == key
                CUDA.atomic_xchg!(pointer(table_values, idx), value)
                return
            end
            slot = (slot + UInt32(1)) & (K_HASH_TABLE_CAPACITY - UInt32(1))
        end
    end
    return
end

function hashtable_delete_kernel!(table_keys, table_values, keys, num_kvs::UInt32)
    tid = UInt32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
    if tid <= num_kvs
        key = keys[tid]
        slot = hash_key(key)

        while true
            idx = slot + UInt32(1)
            if table_keys[idx] == key
                table_values[idx] = K_EMPTY
                return
            end
            if table_keys[idx] == K_EMPTY
                return
            end
            slot = (slot + UInt32(1)) & (K_HASH_TABLE_CAPACITY - UInt32(1))
        end
    end
    return
end

function iterate_hashtable_kernel!(table_keys, table_values, out_keys, out_values, out_size)
    tid = UInt32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
    if tid < K_HASH_TABLE_CAPACITY
        idx = tid + UInt32(1)
        key = table_keys[idx]
        if key != K_EMPTY
            value = table_values[idx]
            if value != K_EMPTY
                pos = CUDA.atomic_add!(pointer(out_size, 1), UInt32(1)) + UInt32(1)
                out_keys[pos] = key
                out_values[pos] = value
            end
        end
    end
    return
end

function generate_random_keyvalues(rng::AbstractRNG, n::Int)
    keys = Vector{UInt32}(undef, n)
    values = Vector{UInt32}(undef, n)
    for i in 1:n
        keys[i] = rand(rng, UInt32) % K_EMPTY
        values[i] = rand(rng, UInt32) % K_EMPTY
    end
    return keys, values
end

function shuffle_keyvalues(rng::AbstractRNG, keys::Vector{UInt32}, values::Vector{UInt32}, n::Int)
    idxs = collect(UInt32, UInt32(1):UInt32(length(keys)))
    shuffle!(rng, idxs)
    delete_keys = Vector{UInt32}(undef, n)
    delete_values = Vector{UInt32}(undef, n)
    for i in 1:n
        idx = Int(idxs[i])
        delete_keys[i] = keys[idx]
        delete_values[i] = values[idx]
    end
    return delete_keys, delete_values
end

function insert_hashtable!(table_keys, table_values, keys, values, num_kvs::Int)
    blocks = cld(num_kvs, THREADS)
    CUDA.synchronize()
    start = time_ns()
    @cuda blocks=blocks threads=THREADS hashtable_insert_kernel!(
        table_keys, table_values, keys, values, UInt32(num_kvs))
    CUDA.synchronize()
    return time_ns() - start
end

function delete_hashtable!(table_keys, table_values, keys, num_kvs::Int)
    blocks = cld(num_kvs, THREADS)
    CUDA.synchronize()
    start = time_ns()
    @cuda blocks=blocks threads=THREADS hashtable_delete_kernel!(
        table_keys, table_values, keys, UInt32(num_kvs))
    CUDA.synchronize()
    return time_ns() - start
end

function iterate_hashtable(table_keys, table_values)
    out_size = CUDA.zeros(UInt32, 1)
    out_keys = CUDA.zeros(UInt32, Int(K_NUM_KEYVALUES))
    out_values = CUDA.zeros(UInt32, Int(K_NUM_KEYVALUES))
    blocks = cld(Int(K_HASH_TABLE_CAPACITY), THREADS)

    CUDA.synchronize()
    start = time_ns()
    @cuda blocks=blocks threads=THREADS iterate_hashtable_kernel!(
        table_keys, table_values, out_keys, out_values, out_size)
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Kernel execution time (iterate): %f (s)\n", elapsed * 1e-9)

    num_kvs = Int(Array(out_size)[1])
    return Array(out_keys)[1:num_kvs], Array(out_values)[1:num_kvs]
end

function test_unordered_map(insert_keys, insert_values, delete_keys)
    start = time_ns()
    println("Timing std::unordered_map...")
    kvs_map = Dict{UInt32,UInt32}()
    for i in eachindex(insert_keys)
        kvs_map[insert_keys[i]] = insert_values[i]
    end
    for key in delete_keys
        delete!(kvs_map, key)
    end
    milliseconds = (time_ns() - start) * 1e-6
    seconds = milliseconds / 1000.0
    @printf("Total time for std::unordered_map: %f ms (%f million keys/second)\n",
            milliseconds, Float64(K_NUM_KEYVALUES) / seconds / 1_000_000.0)
end

function test_correctness(insert_keys, insert_values, delete_keys, kv_keys, kv_values)
    println("Testing that there are no duplicate keys...")
    unique_keys = Set{UInt32}()
    for i in eachindex(kv_keys)
        if (i - 1) % 10_000_000 == 0
            @printf("    Verifying %d/%d\n", i - 1, length(kv_keys))
        end
        key = kv_keys[i]
        if key in unique_keys
            @printf("Duplicate key found in GPU hash table at slot %d\n", i - 1)
            return false
        end
        push!(unique_keys, key)
    end

    println("Building unordered_map from original list...")
    all_kvs_map = Dict{UInt32,Vector{UInt32}}()
    for i in eachindex(insert_keys)
        if (i - 1) % 10_000_000 == 0
            @printf("    Inserting %u/%u\n", i - 1, length(insert_keys))
        end
        values = get!(all_kvs_map, insert_keys[i], UInt32[])
        push!(values, insert_values[i])
    end

    for i in eachindex(delete_keys)
        if (i - 1) % 10_000_000 == 0
            @printf("    Deleting %u/%u\n", i - 1, length(delete_keys))
        end
        delete!(all_kvs_map, delete_keys[i])
    end

    if length(unique_keys) != length(all_kvs_map)
        println("# of unique keys in hashtable is incorrect")
        return false
    end

    println("Testing that each key/value in hashtable is in the original list...")
    for i in eachindex(kv_keys)
        if (i - 1) % 10_000_000 == 0
            @printf("    Verifying %d/%d\n", i - 1, length(kv_keys))
        end
        values = get(all_kvs_map, kv_keys[i], nothing)
        if values === nothing
            println("Hashtable key not found in original list")
            return false
        end
        if !(kv_values[i] in values)
            println("Hashtable value not found in original list")
            return false
        end
    end

    println("Deleting std::unordered_map and std::unique_set...")
    return true
end

function main()
    if length(ARGS) != 2
        @printf("Usage: %s <number of insert batches> <number of delete batches>\n", PROGRAM_FILE)
        return 1
    end

    num_insert_batches = parse(Int, ARGS[1])
    num_delete_batches = parse(Int, ARGS[2])
    seed = 123
    rng = MersenneTwister(seed)

    @printf("Random number generator seed = %u\n", seed)
    println("Initializing keyvalue pairs with random numbers...")
    insert_keys, insert_values = generate_random_keyvalues(rng, Int(K_NUM_KEYVALUES))
    delete_keys, delete_values = shuffle_keyvalues(
        rng, insert_keys, insert_values, Int(K_NUM_KEYVALUES ÷ UInt32(2)))

    @printf("Testing insertion/deletion of %d/%d elements into GPU hash table...\n",
            length(insert_keys), length(delete_keys))

    total_start = time_ns()
    table_keys = CuArray(fill(K_EMPTY, Int(K_HASH_TABLE_CAPACITY)))
    table_values = CuArray(fill(K_EMPTY, Int(K_HASH_TABLE_CAPACITY)))

    total_ktime = 0.0
    num_inserts_per_batch = length(insert_keys) ÷ num_insert_batches
    for i in 0:num_insert_batches-1
        lo = i * num_inserts_per_batch + 1
        hi = lo + num_inserts_per_batch - 1
        d_keys = CuArray(@view insert_keys[lo:hi])
        d_values = CuArray(@view insert_values[lo:hi])
        total_ktime += insert_hashtable!(table_keys, table_values, d_keys, d_values, num_inserts_per_batch)
    end
    @printf("Average kernel execution time (insert): %f (s)\n",
            (total_ktime * 1e-9) / num_insert_batches)

    total_ktime = 0.0
    num_deletes_per_batch = length(delete_keys) ÷ num_delete_batches
    for i in 0:num_delete_batches-1
        lo = i * num_deletes_per_batch + 1
        hi = lo + num_deletes_per_batch - 1
        d_keys = CuArray(@view delete_keys[lo:hi])
        total_ktime += delete_hashtable!(table_keys, table_values, d_keys, num_deletes_per_batch)
    end
    @printf("Average kernel execution time (delete): %f (s)\n",
            (total_ktime * 1e-9) / num_delete_batches)

    kv_keys, kv_values = iterate_hashtable(table_keys, table_values)

    milliseconds = (time_ns() - total_start) * 1e-6
    seconds = milliseconds / 1000.0
    @printf("Total time (including memory copies, readback, etc): %f ms (%f million keys/second)\n",
            milliseconds, Float64(K_NUM_KEYVALUES) / seconds / 1_000_000.0)

    test_unordered_map(insert_keys, insert_values, delete_keys)
    ok = test_correctness(insert_keys, insert_values, delete_keys, kv_keys, kv_values)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
