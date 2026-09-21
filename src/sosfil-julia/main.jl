using CUDA
using Printf

const SOS_WIDTH = 6
const THREADS = 64

function reference!(n_signals::Int, n_samples::Int, n_sections::Int, zi_width::Int,
                    sos::Vector{T}, zi::Vector{T}, x_in::Vector{T}) where {T}
    for ty in 0:n_signals-1
        s_out = zeros(T, n_sections)
        s_zi = zeros(T, n_sections * zi_width)
        s_sos = zeros(T, n_sections * SOS_WIDTH)
        for tx in 0:n_sections-1, i in 0:zi_width-1
            s_zi[tx * zi_width + i + 1] = zi[ty * n_sections * zi_width + tx * zi_width + i + 1]
        end
        copyto!(s_sos, sos)
        load_size = n_sections - 1
        unload_size = n_samples - load_size
        x_n = zeros(T, n_sections)

        for n in 0:load_size-1
            for tx in 0:n_sections-1
                x_n[tx + 1] = tx == 0 ? x_in[ty * n_samples + n + 1] : s_out[tx]
            end
            for tx in 0:n_sections-1
                temp = s_sos[tx * SOS_WIDTH + 1] * x_n[tx + 1] + s_zi[tx * zi_width + 1]
                s_zi[tx * zi_width + 1] = s_sos[tx * SOS_WIDTH + 2] * x_n[tx + 1] -
                                          s_sos[tx * SOS_WIDTH + 5] * temp +
                                          s_zi[tx * zi_width + 2]
                s_zi[tx * zi_width + 2] = s_sos[tx * SOS_WIDTH + 3] * x_n[tx + 1] -
                                          s_sos[tx * SOS_WIDTH + 6] * temp
                s_out[tx + 1] = temp
            end
        end

        for n in load_size:n_samples-1
            for tx in 0:n_sections-1
                x_n[tx + 1] = tx == 0 ? x_in[ty * n_samples + n + 1] : s_out[tx]
            end
            for tx in 0:n_sections-1
                temp = s_sos[tx * SOS_WIDTH + 1] * x_n[tx + 1] + s_zi[tx * zi_width + 1]
                s_zi[tx * zi_width + 1] = s_sos[tx * SOS_WIDTH + 2] * x_n[tx + 1] -
                                          s_sos[tx * SOS_WIDTH + 5] * temp +
                                          s_zi[tx * zi_width + 2]
                s_zi[tx * zi_width + 2] = s_sos[tx * SOS_WIDTH + 3] * x_n[tx + 1] -
                                          s_sos[tx * SOS_WIDTH + 6] * temp
                if tx < load_size
                    s_out[tx + 1] = temp
                else
                    x_in[ty * n_samples + (n - load_size) + 1] = temp
                end
            end
        end

        for n in 0:n_sections-1
            for tx in n+1:n_sections-1
                x_n[tx + 1] = s_out[tx]
            end
            for tx in n+1:n_sections-1
                temp = s_sos[tx * SOS_WIDTH + 1] * x_n[tx + 1] + s_zi[tx * zi_width + 1]
                s_zi[tx * zi_width + 1] = s_sos[tx * SOS_WIDTH + 2] * x_n[tx + 1] -
                                          s_sos[tx * SOS_WIDTH + 5] * temp +
                                          s_zi[tx * zi_width + 2]
                s_zi[tx * zi_width + 2] = s_sos[tx * SOS_WIDTH + 3] * x_n[tx + 1] -
                                          s_sos[tx * SOS_WIDTH + 6] * temp
                if tx < load_size
                    s_out[tx + 1] = temp
                else
                    x_in[ty * n_samples + (n + unload_size) + 1] = temp
                end
            end
        end
    end
    return x_in
end

function compare_results(cpu::Vector{T}, gpu::Vector{T}, atol::T, rtol::T) where {T}
    for i in eachindex(cpu)
        diff = abs(cpu[i] - gpu[i])
        tol = atol + rtol * abs(cpu[i])
        if diff > tol
            @printf("Mismatch at index %d: CPU=%e GPU=%e diff=%e tol=%e\n",
                    i - 1, Float64(cpu[i]), Float64(gpu[i]), Float64(diff), Float64(tol))
            return false
        end
    end
    return true
end

function filtering(::Type{T}, repeat::Int, n_signals::Int, n_samples::Int,
                   n_sections::Int, zi_width::Int) where {T}
    sos = ones(T, n_sections * SOS_WIDTH)
    zi = ones(T, n_sections * n_signals * zi_width)
    x = Vector{T}(undef, n_signals * n_samples)
    x_ref = similar(x)
    for i in 0:n_signals-1, j in 0:n_samples-1
        x[i * n_samples + j + 1] = T(sin(2 * 3.14 * (i + 1 + j)))
    end
    copyto!(x_ref, x)

    for _ in 1:30
        reference!(n_signals, n_samples, n_sections, zi_width, sos, zi, x)
        reference!(n_signals, n_samples, n_sections, zi_width, sos, zi, x_ref)
    end
    ok = compare_results(x_ref, x, T(1e-4), T(1e-4))
    println(ok ? "PASS" : "FAIL")

    start = time()
    for _ in 1:repeat
        reference!(n_signals, n_samples, n_sections, zi_width, sos, zi, x)
    end
    elapsed = time() - start
    @printf("Average kernel execution time %lf (s)\n", elapsed / repeat)
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, ARGS[1])
    n_sections = THREADS
    n_signals = parse(Int, get(ENV, "SOSFIL_SIGNALS", "8"))
    n_samples = parse(Int, get(ENV, "SOSFIL_SAMPLES", "1000000"))
    zi_width = 2
    println("Single-precision second-order-section filtering of digital signals")
    filtering(Float32, repeat, n_signals, n_samples, n_sections, zi_width)
    println("Double-precision second-order-section filtering of digital signals")
    filtering(Float64, repeat, n_signals, n_samples, n_sections, zi_width)
    return 0
end

exit(main())
