using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_bitonic!(i,a,seq_len::Int32,two_power::Int32,n::Int32)
    i0=Int32(i)-Int32(1)
    if i0<n
    seq_num = div(i0, seq_len)
    h_len = div(seq_len, Int32(2))
    swapped = Int32(-1)
    if i0 < (seq_len * seq_num) + h_len
        swapped = i0 + h_len
    end
    odd = div(seq_num, two_power)
    increasing = (odd & Int32(1)) == Int32(0)
    if swapped != Int32(-1)
        # 1-based Julia indexing
        @inbounds ai = a[i0 + Int32(1)]
        @inbounds aj = a[swapped + Int32(1)]
        if (ai > aj && increasing) || (ai < aj && !increasing)
            @inbounds a[i0 + Int32(1)] = aj
            @inbounds a[swapped + Int32(1)] = ai
        end
    end
    end
    nothing
end
function study_prepare(data,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=n÷256,shmem_size=0,sync=false)
    (seq,power)->JACC.parallel_for(portable_bitonic!,spec,n,data,seq,power,Int32(n))
end
exit(study_main(study_prepare))
