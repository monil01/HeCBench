using CUDA,KernelAbstractions,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function attention_ka1!(key, query, dot_product, exp_sum, n::Int32, d::Int32)
    i = Int32(@index(Global,Linear))-Int32(1)
    if i < n
        sum = 0.0f0
        j = Int32(0)
        while j < d
            sum += key[Int64(i) * Int64(d) + Int64(j) + 1] * query[j + Int32(1)]
            j += Int32(1)
        end
        dot_product[i + Int32(1)] = sum
        Atomix.@atomic exp_sum[1] += exp(sum)
    end
end

@kernel function attention_ka2!(exp_sum, dot_product, score, n::Int32)
    i = Int32(@index(Global,Linear))-Int32(1)
    if i < n
        score[i + Int32(1)] = exp(dot_product[i + Int32(1)]) / exp_sum[1]
    end
end

@kernel function attention_ka3!(score, value, output, n::Int32, d::Int32)
    j = Int32(@index(Global,Linear))-Int32(1)
    if j < d
        sum = 0.0f0
        i = Int32(0)
        while i < n
            sum += score[i + Int32(1)] * value[Int64(i) * Int64(d) + Int64(j) + 1]
            i += Int32(1)
        end
        output[j + Int32(1)] = sum
    end
end

function factory(key,value,query,dot,es,score,output)
    k1=attention_ka1!(get_backend(key),256);k2=attention_ka2!(get_backend(key),256);k3=attention_ka3!(get_backend(key),256)
    return ()->begin
        fill!(es,0f0)
        k1(key,query,dot,es,Int32(1024),Int32(64);ndrange=1024)
        k2(es,dot,score,Int32(1024);ndrange=1024)
        k3(score,value,output,Int32(1024),Int32(64);ndrange=64)
    end
end
attention_study(factory)
