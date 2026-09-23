using CUDA
using Printf
using Random

const MAX_MEM = 4

function ccsd_kernel!(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t,
                      dintc1, dintx1, t1v1, dintc2, dintx2, t1v2, eorb,
                      out, eaijk::Float64, ncor::Int32, nocc::Int32, nvir::Int32)
    b = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    c = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if b <= nvir && c <= nvir
        bc = b + (c - Int32(1)) * nvir
        cb = c + (b - Int32(1)) * nvir
        denom = -1.0 / (eorb[ncor + nocc + b] + eorb[ncor + nocc + c] + eaijk)

        f1nbc = f1n[bc]; f1tbc = f1t[bc]; f1ncb = f1n[cb]; f1tcb = f1t[cb]
        f2nbc = f2n[bc]; f2tbc = f2t[bc]; f2ncb = f2n[cb]; f2tcb = f2t[cb]
        f3nbc = f3n[bc]; f3tbc = f3t[bc]; f3ncb = f3n[cb]; f3tcb = f3t[cb]
        f4nbc = f4n[bc]; f4tbc = f4t[bc]; f4ncb = f4n[cb]; f4tcb = f4t[cb]

        s1 = denom * (f1tbc + f1ncb + f2tcb + f3nbc + f4ncb) * (f1tbc - f2tbc * 2 - f3tbc * 2 + f4tbc) -
             denom * (f1nbc + f1tcb + f2ncb + f3ncb) * (f1tbc * 2 - f2tbc - f3tbc + f4tbc * 2) +
             denom * 3 * (f1nbc * (f1nbc + f3ncb + f4tcb * 2) + f2nbc * f2tcb + f3nbc * f4tbc)

        s2 = denom * (f1nbc + f1tcb + f2ncb + f3tbc + f4tcb) * (f1nbc - f2nbc * 2 - f3nbc * 2 + f4nbc) -
             denom * (f1tbc + f1ncb + f2tcb + f3tcb) * (f1nbc * 2 - f2nbc - f3nbc + f4nbc * 2) +
             denom * 3 * (f1tbc * (f1tbc + f3tcb + f4ncb * 2) + f2tbc * f2ncb + f3tbc * f4nbc)

        t1v1b = t1v1[b]; t1v2b = t1v2[b]
        dintx1c = dintx1[c]; dintx2c = dintx2[c]; dintc1c = dintc1[c]; dintc2c = dintc2[c]

        s3 = denom * t1v1b * dintx1c * (f1tbc + f2nbc + f4ncb - (f3tbc + f4nbc + f2ncb + f1nbc + f2tbc + f3ncb) * 2 +
             (f3nbc + f4tbc + f1ncb) * 4) +
             denom * t1v1b * dintc1c * (f1nbc + f4nbc + f1tcb - (f2nbc + f3nbc + f2tcb) * 2)

        s4 = denom * t1v2b * dintx2c * (f1nbc + f2tbc + f4tcb - (f3nbc + f4tbc + f2tcb + f1tbc + f2nbc + f3tcb) * 2 +
             (f3tbc + f4nbc + f1tcb) * 4) +
             denom * t1v2b * dintc2c * (f1tbc + f4tbc + f1ncb - (f2tbc + f3tbc + f2ncb) * 2)

        CUDA.@atomic out[1] += s1
        CUDA.@atomic out[2] += s3
        CUDA.@atomic out[3] += s2
        CUDA.@atomic out[4] += s4
    end
    return
end

function kernel_ref(arrs, eorb, eaijk, ncor, nocc, nvir)
    f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t, dintc1, dintx1, t1v1, dintc2, dintx2, t1v2 = arrs
    emp4i = 0.0; emp5i = 0.0; emp4k = 0.0; emp5k = 0.0
    for c in 1:nvir, b in 1:nvir
        denom = -1.0 / (eorb[ncor + nocc + b] + eorb[ncor + nocc + c] + eaijk)
        bc = b + (c - 1) * nvir
        cb = c + (b - 1) * nvir
        f1nbc = f1n[bc]; f1tbc = f1t[bc]; f1ncb = f1n[cb]; f1tcb = f1t[cb]
        f2nbc = f2n[bc]; f2tbc = f2t[bc]; f2ncb = f2n[cb]; f2tcb = f2t[cb]
        f3nbc = f3n[bc]; f3tbc = f3t[bc]; f3ncb = f3n[cb]; f3tcb = f3t[cb]
        f4nbc = f4n[bc]; f4tbc = f4t[bc]; f4ncb = f4n[cb]; f4tcb = f4t[cb]

        emp4i += denom * (f1tbc + f1ncb + f2tcb + f3nbc + f4ncb) * (f1tbc - f2tbc * 2 - f3tbc * 2 + f4tbc) -
                 denom * (f1nbc + f1tcb + f2ncb + f3ncb) * (f1tbc * 2 - f2tbc - f3tbc + f4tbc * 2) +
                 denom * 3 * (f1nbc * (f1nbc + f3ncb + f4tcb * 2) + f2nbc * f2tcb + f3nbc * f4tbc)
        emp4k += denom * (f1nbc + f1tcb + f2ncb + f3tbc + f4tcb) * (f1nbc - f2nbc * 2 - f3nbc * 2 + f4nbc) -
                 denom * (f1tbc + f1ncb + f2tcb + f3tcb) * (f1nbc * 2 - f2nbc - f3nbc + f4nbc * 2) +
                 denom * 3 * (f1tbc * (f1tbc + f3tcb + f4ncb * 2) + f2tbc * f2ncb + f3tbc * f4nbc)

        t1v1b = t1v1[b]; t1v2b = t1v2[b]
        dintx1c = dintx1[c]; dintx2c = dintx2[c]; dintc1c = dintc1[c]; dintc2c = dintc2[c]
        emp5i += denom * t1v1b * dintx1c * (f1tbc + f2nbc + f4ncb - (f3tbc + f4nbc + f2ncb + f1nbc + f2tbc + f3ncb) * 2 + (f3nbc + f4tbc + f1ncb) * 4) +
                 denom * t1v1b * dintc1c * (f1nbc + f4nbc + f1tcb - (f2nbc + f3nbc + f2tcb) * 2)
        emp5k += denom * t1v2b * dintx2c * (f1nbc + f2tbc + f4tcb - (f3nbc + f4tbc + f2tcb + f1tbc + f2nbc + f3tcb) * 2 + (f3tbc + f4nbc + f1tcb) * 4) +
                 denom * t1v2b * dintc2c * (f1tbc + f4tbc + f1ncb - (f2tbc + f3tbc + f2ncb) * 2)
    end
    return emp4i, emp5i, emp4k, emp5k
end

function run_gpu(cuarrs, d_eorb, eaijk, ncor, nocc, nvir)
    out = CUDA.zeros(Float64, 4)
    threads = (16, 16)
    blocks = (cld(nvir, 16), cld(nvir, 16))
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=threads blocks=blocks ccsd_kernel!(cuarrs..., d_eorb, out, eaijk, Int32(ncor), Int32(nocc), Int32(nvir))
    CUDA.synchronize()
    return Array(out), time_ns() - t0
end

function wrap_energy(x)
    if x > 1000.0
        return x - 1000.0
    elseif x < -1000.0
        return x + 1000.0
    end
    return x
end

function main(args)
    if length(args) < 2
        println("Usage: ./$(basename(PROGRAM_FILE)) nocc nvir [maxiter] [nkpass]")
        return length(args)
    end
    ncor = 0
    nocc = parse(Int, args[1])
    nvir = parse(Int, args[2])
    maxiter = length(args) > 2 ? parse(Int, args[3]) : 100
    maxiter = maxiter < 0 ? (1 << 30) : maxiter
    nkpass = length(args) > 3 ? parse(Int, args[4]) : 1
    if nocc < 1 || nvir < 1
        println("Arguments must be non-negative!")
        return 1
    end

    println("Test driver for cbody with nocc=$(nocc), nvir=$(nvir), maxiter=$(maxiter), nkpass=$(nkpass)")
    nbf = ncor + nocc + nvir
    lnvv = nvir * nvir
    lnov = nocc * nvir
    kchunk = (nocc - 1) ÷ nkpass + 1
    memory = (nbf + 8.0 * lnvv + lnvv + kchunk * lnvv + lnov * nocc + kchunk * lnov +
              lnov * nocc + kchunk * lnov + lnvv + kchunk * lnvv + lnvv + kchunk * lnvv +
              lnov * nocc + kchunk * lnov + lnov * nocc + kchunk * lnov + lnov +
              nvir * kchunk + nvir * nocc + 6.0 * lnvv) * sizeof(Float64)
    @printf("This test requires %f GB of memory.\n", 1.0e-9 * memory)
    if 1.0e-9 * memory > MAX_MEM
        println("You need to increase MAX_MEM ($(MAX_MEM))")
        println("or set nkpass ($(nkpass)) to a larger number.")
        return MAX_MEM
    end

    rng = MersenneTwister(2)
    make_array(n) = rand(rng, Float64, n)
    eorb = make_array(nbf)
    arrs = ntuple(_ -> make_array(lnvv), 8)
    scratch = ntuple(_ -> make_array(nvir), 6)
    allarrs = (arrs..., scratch...)
    cuarrs = map(CuArray, allarrs)
    d_eorb = CuArray(eorb)

    ntimers = min(maxiter, nocc * nocc * nocc * nocc)
    timers = zeros(Float64, ntimers)
    emp4 = 0.0; emp5 = 0.0; emp4_r = 0.0; emp5_r = 0.0
    iter = 0

    for klo in 1:kchunk:nocc
        khi = min(nocc, klo + kchunk - 1)
        a = 1
        for j in 1:nocc, i in 1:nocc, k in klo:min(khi, i)
            eaijk = eorb[a] - (eorb[ncor + i] + eorb[ncor + j] + eorb[ncor + k])
            out, t = run_gpu(cuarrs, d_eorb, eaijk, ncor, nocc, nvir)
            emp4 += out[1]; emp5 += out[2]
            if i != k
                emp4 += out[3]; emp5 += out[4]
            end
            r = kernel_ref(allarrs, eorb, eaijk, ncor, nocc, nvir)
            emp4_r += r[1]; emp5_r += r[2]
            if i != k
                emp4_r += r[3]; emp5_r += r[4]
            end
            iter += 1
            timers[iter] = t * 1e-9
            if iter == maxiter
                println("Stopping after $(iter) iterations...")
                @goto maxed_out
            end
            emp4 = wrap_energy(emp4)
            emp5 = wrap_energy(emp5)
            emp4_r = wrap_energy(emp4_r)
            emp5_r = wrap_energy(emp5_r)
        end
    end

    @label maxed_out
    used = timers[1:iter]
    tmin = minimum(used); tmax = maximum(used); tavg = sum(used) / iter
    @printf("Kernel timing: min=%lf, max=%lf, avg=%lf\n", tmin, tmax, tavg)
    dgemm_flops = ((8.0 * nvir) * nvir) * (nvir + nocc)
    dgemm_mops = 8.0 * (4.0 * nvir * nvir + 2.0 * nvir * nocc)
    tengy_ops = (1.0 * nvir * nvir) * (86 + 8)
    @printf("OPS: dgemm_flops=%10.3e dgemm_mops=%10.3e tengy_ops=%10.3e\n", dgemm_flops, dgemm_mops, tengy_ops)
    @printf("PERF: GF/s=%10.3e GB/s=%10.3e\n", 1.0e-9 * (dgemm_flops + tengy_ops) / tavg, 8.0e-9 * (dgemm_mops + tengy_ops) / tavg)
    println("These are meaningless but should not vary for a particular input:")
    @printf("emp4=%f emp5=%f\n", emp4, emp5)
    println(abs(emp4_r - emp4) < 1e-4 && abs(emp5_r - emp5) < 1e-4 ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
