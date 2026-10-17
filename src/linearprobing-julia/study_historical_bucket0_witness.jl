using CUDA
const K_HASH_TABLE_CAPACITY=UInt32(64*1024*1024)
const K_EMPTY=typemax(UInt32)
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

# Partial-launchminimalcounterexample:1blockcoversatmost257slots, nofullCLI/performanceclaim.
keys=CuArray(fill(K_EMPTY,257));values=CuArray(fill(K_EMPTY,257));copyto!(view(keys,1:1),UInt32[0]);copyto!(view(values,1:1),UInt32[123]);outk=CUDA.zeros(UInt32,2);outv=CUDA.zeros(UInt32,2);count=CUDA.zeros(UInt32,1)
@cuda threads=256 blocks=1 iterate_hashtable_kernel!(keys,values,outk,outv,count)
CUDA.synchronize();got=Array(count)[1];println("WITNESS model=historical_cuda_jl expected_count=1 actual_count=",got," table_slot0_key=0 table_slot0_value=123 original_kernel_fragment_sha256=1a561fffbdcfbd04bd5a45579bb25d480b9976371bf838fc6f337d00c99b31bd ")
got==0||error("counterexampleunexpected")
