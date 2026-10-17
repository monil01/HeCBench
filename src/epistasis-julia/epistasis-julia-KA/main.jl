using CUDA, KernelAbstractions
using Printf
using Random

const BLOCK_SNP = 64

@inline function gammafunction(n::UInt32)
    n == 0x00000000 && return 0.0f0
    nf = Float32(n)
    return (nf + 0.5f0) * log(nf) - (nf - 1.0f0)
end

@kernel function epi_ka!(data_zeros, data_ones, scores, num_snp::Int32,
                     pp_zeros::Int32, pp_ones::Int32,
                     mask_zeros::UInt32, mask_ones::UInt32)
    indices = @index(Global, NTuple)
    j = Int32(indices[1])
    i = Int32(indices[2])
    if j > i && i <= num_snp && j <= num_snp
        ft1 = ntuple(_ -> UInt32(0), 9)
        ft2 = ntuple(_ -> UInt32(0), 9)

        for p in Int32(0):(Int32(2) * num_snp):(Int32(2) * pp_zeros * num_snp - Int32(2) * num_snp - Int32(1))
            ip = (p + (i - Int32(1)) * Int32(2)) + Int32(1)
            jp = (p + (j - Int32(1)) * Int32(2)) + Int32(1)
            si0 = @inbounds data_zeros[ip]
            si1 = @inbounds data_zeros[ip + Int32(1)]
            sj0 = @inbounds data_zeros[jp]
            sj1 = @inbounds data_zeros[jp + Int32(1)]
            di2 = ~(si0 | si1)
            dj2 = ~(sj0 | sj1)
            ft1 = (ft1[1] + UInt32(count_ones(si0 & sj0)),
                   ft1[2] + UInt32(count_ones(si0 & sj1)),
                   ft1[3] + UInt32(count_ones(si0 & dj2)),
                   ft1[4] + UInt32(count_ones(si1 & sj0)),
                   ft1[5] + UInt32(count_ones(si1 & sj1)),
                   ft1[6] + UInt32(count_ones(si1 & dj2)),
                   ft1[7] + UInt32(count_ones(di2 & sj0)),
                   ft1[8] + UInt32(count_ones(di2 & sj1)),
                   ft1[9] + UInt32(count_ones(di2 & dj2)))
        end

        p = Int32(2) * pp_zeros * num_snp - Int32(2) * num_snp
        ip = (p + (i - Int32(1)) * Int32(2)) + Int32(1)
        jp = (p + (j - Int32(1)) * Int32(2)) + Int32(1)
        si0 = @inbounds data_zeros[ip]
        si1 = @inbounds data_zeros[ip + Int32(1)]
        sj0 = @inbounds data_zeros[jp]
        sj1 = @inbounds data_zeros[jp + Int32(1)]
        di2 = ~(si0 | si1) & mask_zeros
        dj2 = ~(sj0 | sj1) & mask_zeros
        ft1 = (ft1[1] + UInt32(count_ones(si0 & sj0)),
               ft1[2] + UInt32(count_ones(si0 & sj1)),
               ft1[3] + UInt32(count_ones(si0 & dj2)),
               ft1[4] + UInt32(count_ones(si1 & sj0)),
               ft1[5] + UInt32(count_ones(si1 & sj1)),
               ft1[6] + UInt32(count_ones(si1 & dj2)),
               ft1[7] + UInt32(count_ones(di2 & sj0)),
               ft1[8] + UInt32(count_ones(di2 & sj1)),
               ft1[9] + UInt32(count_ones(di2 & dj2)))

        for p in Int32(0):(Int32(2) * num_snp):(Int32(2) * pp_ones * num_snp - Int32(2) * num_snp - Int32(1))
            ip = (p + (i - Int32(1)) * Int32(2)) + Int32(1)
            jp = (p + (j - Int32(1)) * Int32(2)) + Int32(1)
            si0 = @inbounds data_ones[ip]
            si1 = @inbounds data_ones[ip + Int32(1)]
            sj0 = @inbounds data_ones[jp]
            sj1 = @inbounds data_ones[jp + Int32(1)]
            di2 = ~(si0 | si1)
            dj2 = ~(sj0 | sj1)
            ft2 = (ft2[1] + UInt32(count_ones(si0 & sj0)),
                   ft2[2] + UInt32(count_ones(si0 & sj1)),
                   ft2[3] + UInt32(count_ones(si0 & dj2)),
                   ft2[4] + UInt32(count_ones(si1 & sj0)),
                   ft2[5] + UInt32(count_ones(si1 & sj1)),
                   ft2[6] + UInt32(count_ones(si1 & dj2)),
                   ft2[7] + UInt32(count_ones(di2 & sj0)),
                   ft2[8] + UInt32(count_ones(di2 & sj1)),
                   ft2[9] + UInt32(count_ones(di2 & dj2)))
        end

        p = Int32(2) * pp_ones * num_snp - Int32(2) * num_snp
        ip = (p + (i - Int32(1)) * Int32(2)) + Int32(1)
        jp = (p + (j - Int32(1)) * Int32(2)) + Int32(1)
        si0 = @inbounds data_ones[ip]
        si1 = @inbounds data_ones[ip + Int32(1)]
        sj0 = @inbounds data_ones[jp]
        sj1 = @inbounds data_ones[jp + Int32(1)]
        di2 = ~(si0 | si1) & mask_ones
        dj2 = ~(sj0 | sj1) & mask_ones
        ft2 = (ft2[1] + UInt32(count_ones(si0 & sj0)),
               ft2[2] + UInt32(count_ones(si0 & sj1)),
               ft2[3] + UInt32(count_ones(si0 & dj2)),
               ft2[4] + UInt32(count_ones(si1 & sj0)),
               ft2[5] + UInt32(count_ones(si1 & sj1)),
               ft2[6] + UInt32(count_ones(si1 & dj2)),
               ft2[7] + UInt32(count_ones(di2 & sj0)),
               ft2[8] + UInt32(count_ones(di2 & sj1)),
               ft2[9] + UInt32(count_ones(di2 & dj2)))

        score = 0.0f0
        Base.Cartesian.@nexprs 9 k -> begin
            score += gammafunction(ft1[k] + ft2[k] + UInt32(1)) - gammafunction(ft1[k]) - gammafunction(ft2[k])
        end
        score = abs(score)
        if score == 0.0f0
            score = floatmax(Float32)
        end
        @inbounds scores[(i - Int32(1)) * num_snp + j] = score
    end
end


include(joinpath(@__DIR__,"..","study_common.jl"))
kernel=epi_ka!(CUDA.CUDABackend(),(64,1))
study((z,o,s)->kernel(z,o,s,Int32(67),Int32(3),Int32(3),UInt32(7),UInt32(15);ndrange=(128,128)))
