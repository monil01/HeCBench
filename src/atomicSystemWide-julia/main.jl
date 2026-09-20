using CUDA
using Printf

function atomic_kernel!(atom_arr, loop_num::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    for _ in Int32(0):(loop_num - Int32(1))
        CUDA.atomic_add!(pointer(atom_arr, 1), Int32(10))
        CUDA.atomic_xchg!(pointer(atom_arr, 2), tid)
        CUDA.atomic_max!(pointer(atom_arr, 3), tid)
        CUDA.atomic_min!(pointer(atom_arr, 4), tid)
        CUDA.atomic_cas!(pointer(atom_arr, 7), tid - Int32(1), tid)
        CUDA.atomic_and!(pointer(atom_arr, 8), Int32(2) * tid + Int32(7))
        CUDA.atomic_or!(pointer(atom_arr, 9), Int32(1) << tid)
        CUDA.atomic_xor!(pointer(atom_arr, 10), tid)
    end
    return
end

function atomic_kernel_cpu!(atom_arr, no_of_threads, loop_num)
    for i in no_of_threads:(2 * no_of_threads - 1)
        for _ in 1:loop_num
            atom_arr[1] += 10
            atom_arr[2] = i
            atom_arr[3] = max(atom_arr[3], i)
            atom_arr[4] = min(atom_arr[4], i)
            if atom_arr[7] == i - 1
                atom_arr[7] = i
            end
            atom_arr[8] &= 2 * i + 7
            atom_arr[9] |= Int32(1) << i
            atom_arr[10] ⊻= i
        end
    end
end

function verify(test_data, len, loop_num)
    val = 0
    for _ in 1:(len * loop_num)
        val += 10
    end
    if val != test_data[1]
        @printf("atomicAdd failed val = %d testData = %d\n", val, test_data[1])
        return false
    end

    if !(0 <= test_data[2] < len)
        println("atomicExch failed")
        return false
    end

    val = -(1 << 8)
    for i in 0:(len - 1)
        val = max(val, i)
    end
    if val != test_data[3]
        println("atomicMax failed")
        return false
    end

    val = 1 << 8
    for i in 0:(len - 1)
        val = min(val, i)
    end
    if val != test_data[4]
        println("atomicMin failed")
        return false
    end

    if !(0 <= test_data[7] < len)
        println("atomicCAS failed")
        return false
    end

    val = 0xff
    for i in 0:(len - 1)
        val &= 2 * i + 7
    end
    if val != test_data[8]
        println("atomicAnd failed")
        return false
    end

    val = 0
    for i in 0:(len - 1)
        val |= Int32(1) << i
    end
    if val != test_data[9]
        println("atomicOr failed")
        return false
    end

    val = 0xff
    for i in 0:(len - 1)
        val ⊻= i
    end
    if val != test_data[10]
        println("atomicXor failed")
        return false
    end
    return true
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <loop count within the kernel>")
        return 1
    end
    loop_num = parse(Int, args[1])
    num_threads = 256
    num_blocks = 64
    num_data = 10

    println("CAN access pageable memory")
    atom_arr = zeros(Int32, num_data)
    atom_arr[8] = 0xff
    atom_arr[10] = 0xff

    d_atom_arr = CuArray(atom_arr)
    CUDA.synchronize()
    start = time_ns()
    @cuda threads=num_threads blocks=num_blocks atomic_kernel!(d_atom_arr, Int32(loop_num))
    CUDA.synchronize()
    atom_arr .= Array(d_atom_arr)
    atomic_kernel_cpu!(atom_arr, num_blocks * num_threads, loop_num)
    elapsed = time_ns() - start
    @printf("Execution time of atomic kernels on host and device: %f (s)\n", elapsed * 1e-9)

    test_result = verify(atom_arr, 2 * num_threads * num_blocks, loop_num)
    @printf("systemWideAtomics completed, returned %s \n", test_result ? "OK" : "ERROR!")
    return test_result ? 0 : 1
end

exit(main(ARGS))
