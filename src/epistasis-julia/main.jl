using CUDA
using Printf
using Random

const BLOCK_SNP = 64

@inline function gammafunction(n::UInt32)
    n == 0x00000000 && return 0.0f0
    nf = Float32(n)
    return (nf + 0.5f0) * log(nf) - (nf - 1.0f0)
end

function epi_kernel!(data_zeros, data_ones, scores, num_snp::Int32,
                     pp_zeros::Int32, pp_ones::Int32,
                     mask_zeros::UInt32, mask_ones::UInt32)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i = blockIdx().y
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
            score = typemax(Float32)
        end
        @inbounds scores[(i - Int32(1)) * num_snp + j] = score
    end
    return
end

function reference!(data_zeros, data_ones, scores, num_snp, pp_zeros, pp_ones, mask_zeros, mask_ones)
    for i in 1:num_snp, j in 1:num_snp
        j > i || continue
        ft = zeros(UInt32, 18)
        for (data, pp, mask, offset) in ((data_zeros, pp_zeros, mask_zeros, 0), (data_ones, pp_ones, mask_ones, 9))
            for pack in 1:pp
                si0 = data[((pack - 1) * num_snp + (i - 1)) * 2 + 1]
                si1 = data[((pack - 1) * num_snp + (i - 1)) * 2 + 2]
                sj0 = data[((pack - 1) * num_snp + (j - 1)) * 2 + 1]
                sj1 = data[((pack - 1) * num_snp + (j - 1)) * 2 + 2]
                di2 = ~(si0 | si1)
                dj2 = ~(sj0 | sj1)
                if pack == pp
                    di2 &= mask
                    dj2 &= mask
                end
                vals = (si0 & sj0, si0 & sj1, si0 & dj2,
                        si1 & sj0, si1 & sj1, si1 & dj2,
                        di2 & sj0, di2 & sj1, di2 & dj2)
                for k in 1:9
                    ft[offset + k] += UInt32(count_ones(vals[k]))
                end
            end
        end
        score = 0.0f0
        for k in 1:9
            score += gammafunction(ft[k] + ft[9 + k] + UInt32(1)) - gammafunction(ft[k]) - gammafunction(ft[9 + k])
        end
        score = abs(score)
        scores[(i - 1) * num_snp + j] = score == 0.0f0 ? typemax(Float32) : score
    end
    return scores
end

function prepare_inputs(num_pac, num_snp)
    rng = Random.MersenneTwister(100)
    snp_data = rand(rng, UInt8(0):UInt8(2), num_pac, num_snp)
    ph_data = rand(rng, UInt8(0):UInt8(1), num_pac)
    phen_ones = count(==(UInt8(1)), ph_data)
    pp_zeros = cld(num_pac - phen_ones, 32)
    pp_ones = cld(phen_ones, 32)
    bin_zeros = zeros(UInt32, num_snp * pp_zeros * 2)
    bin_ones = zeros(UInt32, num_snp * pp_ones * 2)
    for i in 1:num_snp
        x_zeros = 0
        x_ones = 0
        n_zeros = 0
        n_ones = 0
        for j in 1:num_pac
            temp = snp_data[j, i]
            if ph_data[j] == UInt8(1)
                if n_ones % 32 == 0
                    x_ones += 1
                end
                base = ((i - 1) * pp_ones + (x_ones - 1)) * 2
                bin_ones[base + 1] <<= 1
                bin_ones[base + 2] <<= 1
                if temp <= UInt8(1)
                    bin_ones[base + Int(temp) + 1] |= UInt32(1)
                end
                n_ones += 1
            else
                if n_zeros % 32 == 0
                    x_zeros += 1
                end
                base = ((i - 1) * pp_zeros + (x_zeros - 1)) * 2
                bin_zeros[base + 1] <<= 1
                bin_zeros[base + 2] <<= 1
                if temp <= UInt8(1)
                    bin_zeros[base + Int(temp) + 1] |= UInt32(1)
                end
                n_zeros += 1
            end
        end
    end

    mask_zeros = typemax(UInt32)
    for _ in (num_pac - phen_ones):(pp_zeros * 32 - 1)
        mask_zeros >>= 1
    end
    mask_ones = typemax(UInt32)
    for _ in phen_ones:(pp_ones * 32 - 1)
        mask_ones >>= 1
    end

    trans_zeros = similar(bin_zeros)
    for i in 1:num_snp, j in 1:pp_zeros
        trans_zeros[((j - 1) * num_snp + (i - 1)) * 2 + 1] = bin_zeros[((i - 1) * pp_zeros + (j - 1)) * 2 + 1]
        trans_zeros[((j - 1) * num_snp + (i - 1)) * 2 + 2] = bin_zeros[((i - 1) * pp_zeros + (j - 1)) * 2 + 2]
    end
    trans_ones = similar(bin_ones)
    for i in 1:num_snp, j in 1:pp_ones
        trans_ones[((j - 1) * num_snp + (i - 1)) * 2 + 1] = bin_ones[((i - 1) * pp_ones + (j - 1)) * 2 + 1]
        trans_ones[((j - 1) * num_snp + (i - 1)) * 2 + 2] = bin_ones[((i - 1) * pp_ones + (j - 1)) * 2 + 2]
    end
    return trans_zeros, trans_ones, pp_zeros, pp_ones, mask_zeros, mask_ones
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of samples> <number of SNPs> <repeat>")
        return 1
    end
    num_pac = parse(Int, args[1])
    num_snp = parse(Int, args[2])
    iteration = parse(Int, args[3])

    data_zeros, data_ones, pp_zeros, pp_ones, mask_zeros, mask_ones = prepare_inputs(num_pac, num_snp)
    scores = fill(typemax(Float32), num_snp * num_snp)
    d_zeros = CuArray(data_zeros)
    d_ones = CuArray(data_ones)
    d_scores = CuArray(scores)
    num_snp_m = cld(num_snp, BLOCK_SNP) * BLOCK_SNP
    blocks = (cld(num_snp_m, BLOCK_SNP), num_snp)

    CUDA.synchronize()
    start = time()
    for _ in 1:iteration
        @cuda threads=BLOCK_SNP blocks=blocks epi_kernel!(d_zeros, d_ones, d_scores, Int32(num_snp),
                                                          Int32(pp_zeros), Int32(pp_ones),
                                                          mask_zeros, mask_ones)
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %.9f (s)\n", (time() - start) / iteration)

    scores = Array(d_scores)
    scores_ref = fill(typemax(Float32), num_snp * num_snp)
    reference!(data_zeros, data_ones, scores_ref, num_snp, pp_zeros, pp_ones, mask_zeros, mask_ones)
    p1 = argmin(scores)
    p2 = argmin(scores_ref)
    ok = p1 == p2 && abs(scores[p1] - scores_ref[p2]) < 1.0f-3
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 2
end

exit(main(ARGS))
