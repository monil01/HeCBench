using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function attention_jacc1!(index,key, query, dot_product, exp_sum, n::Int32, d::Int32)
    i = Int32(index)-Int32(1)
    if i < n
        sum = 0.0f0
        j = Int32(0)
        while j < d
            sum += key[Int64(i) * Int64(d) + Int64(j) + 1] * query[j + Int32(1)]
            j += Int32(1)
        end
        dot_product[i + Int32(1)] = sum
        JACC.@atomic exp_sum[1] += exp(sum)
    end
    return
end

function attention_jacc2!(index,exp_sum, dot_product, score, n::Int32)
    i = Int32(index)-Int32(1)
    if i < n
        score[i + Int32(1)] = exp(dot_product[i + Int32(1)]) / exp_sum[1]
    end
    return
end

function attention_jacc3!(index,score, value, output, n::Int32, d::Int32)
    j = Int32(index)-Int32(1)
    if j < d
        sum = 0.0f0
        i = Int32(0)
        while i < n
            sum += score[i + Int32(1)] * value[Int64(i) * Int64(d) + Int64(j) + 1]
            i += Int32(1)
        end
        output[j + Int32(1)] = sum
    end
    return
end

function factory(key,value,query,dot,es,score,output)
    spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    small=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    return ()->begin
        fill!(es,0f0)
        JACC.parallel_for(spec,1024,attention_jacc1!,key,query,dot,es,Int32(1024),Int32(64))
        JACC.parallel_for(spec,1024,attention_jacc2!,es,dot,score,Int32(1024))
        JACC.parallel_for(small,64,attention_jacc3!,score,value,output,Int32(1024),Int32(64))
    end
end
attention_study(factory)
