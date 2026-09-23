using CUDA
using Printf
using Random

mutable struct InputConfig
    source_2d_regions::Int32
    source_3d_regions::Int32
    coarse_axial_intervals::Int32
    fine_axial_intervals::Int32
    decomp_assemblies_ax::Int32
    segments::Int64
    egroups::Int32
    repeat::Int32
    nbytes::Int64
end

function default_input()
    cfg = InputConfig(Int32(5000), Int32(0), Int32(27), Int32(5), Int32(20),
                      50_000_000, Int32(128), Int32(1), 0)
    return cfg
end

function read_cli!(cfg::InputConfig, args)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-s"
            i += 1; cfg.segments = parse(Int64, args[i])
        elseif arg == "-e"
            i += 1; cfg.egroups = Int32(parse(Int, args[i]))
        elseif arg == "-n"
            i += 1; cfg.repeat = Int32(parse(Int, args[i]))
        elseif arg == "-t"
            i += 1
        else
            println("Usage: ./SimpleMOC <options>")
            exit(1)
        end
        i += 1
    end
    cfg.source_3d_regions = Int32(cld(Int(cfg.source_2d_regions) * Int(cfg.coarse_axial_intervals),
                                      Int(cfg.decomp_assemblies_ax)))
    return cfg
end

function fancy_int(n::Integer)
    s = string(n)
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    pushfirst!(parts, s)
    return join(parts, ",")
end

border_print() = println("================================================================================")

function center_print(s)
    width = 79
    n = max(0, (width - length(s)) ÷ 2 + 1)
    println(repeat(" ", n), s)
end

function logo()
    border_print()
    println("   __           __        ___        __   __           ___  __        ___     ")
    println("  /__` |  |\\/| |__) |    |__   |\\/| /  \\ /  ` __ |__/ |__  |__) |\\ | |__  |   ")
    println("  .__/ |  |  | |    |___ |___  |  | \\__/ \\__,    |  \\ |___ |  \\ | \\| |___ |___")
    println()
    border_print()
    println()
    center_print("Developed at")
    center_print("The Massachusetts Institute of Technology")
    center_print("and")
    center_print("Argonne National Laboratory")
    println()
    center_print("Version: 4")
    println()
    border_print()
end

function print_input_summary(cfg::InputConfig)
    center_print("INPUT SUMMARY")
    border_print()
    @printf("%-25s%d\n", "Kernel execution times:", cfg.repeat)
    @printf("%-25s%d\n", "Energy Groups:", cfg.egroups)
    @printf("%-25s%d\n", "2D Source Regions:", cfg.source_2d_regions)
    @printf("%-25s%d\n", "Coarse Axial Intervals:", cfg.coarse_axial_intervals)
    @printf("%-25s%d\n", "Fine Axial Intervals:", cfg.fine_axial_intervals)
    @printf("%-25s%d\n", "Axial Decomposition:", cfg.decomp_assemblies_ax)
    @printf("%-25s%d\n", "3D Source Regions:", cfg.source_3d_regions)
    @printf("%-25s%s\n", "Segments:", fancy_int(cfg.segments))
    @printf("%-25s%.2f\n", "Memory Estimate (MB):", cfg.nbytes / 1024.0 / 1024.0)
    border_print()
end

function attenuation_kernel!(qsr_ids, fai_ids, fine_flux, fine_source, sigt, state_flux,
                             fine_axial_intervals::Int32, egroups::Int32, segments::Int64)
    gid0 = Int64((blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1)))
    if gid0 >= segments
        return
    end

    qsr = qsr_ids[gid0 + 1]
    fai = fai_ids[gid0 + 1]
    dz = 0.1f0
    zin = 0.3f0
    weight = 0.5f0
    mu = 0.9f0
    mu2 = 0.3f0
    ds = 0.7f0

    region_base = qsr * fine_axial_intervals * egroups
    sigt_base = qsr * egroups
    flux_base = region_base + fai * egroups

    for g0 in Int32(0):(egroups - Int32(1))
        f2 = fine_source[region_base + fai * egroups + g0 + Int32(1)]
        q0 = 0.0f0
        q1 = 0.0f0
        q2 = 0.0f0
        if fai == Int32(0)
            f3 = fine_source[region_base + (fai + Int32(1)) * egroups + g0 + Int32(1)]
            c0 = f2
            c1 = (f3 - f2) / dz
            q0 = c0 + c1 * zin
            q1 = c1
        elseif fai == fine_axial_intervals - Int32(1)
            f1 = fine_source[region_base + (fai - Int32(1)) * egroups + g0 + Int32(1)]
            c0 = f2
            c1 = (f2 - f1) / dz
            q0 = c0 + c1 * zin
            q1 = c1
        else
            f1 = fine_source[region_base + (fai - Int32(1)) * egroups + g0 + Int32(1)]
            f3 = fine_source[region_base + (fai + Int32(1)) * egroups + g0 + Int32(1)]
            c0 = f2
            c1 = (f1 - f3) / (2.0f0 * dz)
            c2 = (f1 - 2.0f0 * f2 + f3) / (2.0f0 * dz * dz)
            q0 = c0 + c1 * zin + c2 * zin * zin
            q1 = c1 + 2.0f0 * c2 * zin
            q2 = c2
        end

        st = sigt[sigt_base + g0 + Int32(1)]
        tau = st * ds
        st2 = st * st
        expv = 1.0f0 - exp(-tau)
        reuse = tau * (tau - 2.0f0) + 2.0f0 * expv / (st * st2)
        sf = state_flux[g0 + Int32(1)]
        flux_integral = (q0 * tau + (st * sf - q0) * expv) / st2 +
                        q1 * mu * reuse +
                        q2 * mu2 * (tau * (tau * (tau - 3.0f0) + 6.0f0) - 6.0f0 * expv) /
                        (3.0f0 * st2 * st2)
        fine_flux[flux_base + g0 + Int32(1)] += weight * flux_integral
        t1 = q0 * expv / st
        t2 = q1 * mu * (tau - expv) / st2
        t3 = q2 * mu2 * reuse
        t4 = sf * (1.0f0 - expv)
        state_flux[g0 + Int32(1)] = t1 + t2 + t3 + t4
    end
    return
end

function initialize_data(cfg::InputConfig)
    rng = MersenneTwister(2)
    nsource = Int(cfg.source_3d_regions) * Int(cfg.fine_axial_intervals) * Int(cfg.egroups)
    nsigt = Int(cfg.source_3d_regions) * Int(cfg.egroups)
    fine_source = rand(rng, Float32, nsource)
    fine_flux = rand(rng, Float32, nsource)
    sigt = rand(rng, Float32, nsigt)
    state_flux = rand(rng, Float32, Int(cfg.egroups))
    qsr = Vector{Int32}(undef, cfg.segments)
    fai = Vector{Int32}(undef, cfg.segments)
    for i in eachindex(qsr)
        qsr[i] = Int32(rand(rng, 0:Int(cfg.source_3d_regions)-1))
        fai[i] = Int32(rand(rng, 0:Int(cfg.fine_axial_intervals)-1))
    end
    return fine_source, fine_flux, sigt, state_flux, qsr, fai
end

function main()
    cfg = read_cli!(default_input(), ARGS)
    cfg.nbytes = Int64(cfg.source_3d_regions) * 24 +
                 2 * Int64(cfg.source_3d_regions) * cfg.fine_axial_intervals * cfg.egroups * 4 +
                 Int64(cfg.source_3d_regions) * cfg.egroups * 4

    logo()
    fine_source, fine_flux, sigt, state_flux, qsr, fai = initialize_data(cfg)
    print_input_summary(cfg)

    center_print("SIMULATION")
    border_print()
    println("Attentuating fluxes across segments...")

    start = time_ns()
    d_qsr = CuArray(qsr)
    d_fai = CuArray(fai)
    d_fine_flux = CuArray(fine_flux)
    d_fine_source = CuArray(fine_source)
    d_sigt = CuArray(sigt)
    d_state_flux = CuArray(state_flux)

    threads = 128
    blocks = cld(cfg.segments, threads)
    CUDA.synchronize()
    kstart = time_ns()
    for _ in 1:Int(cfg.repeat)
        @cuda threads=threads blocks=blocks attenuation_kernel!(
            d_qsr, d_fai, d_fine_flux, d_fine_source, d_sigt, d_state_flux,
            cfg.fine_axial_intervals, cfg.egroups, cfg.segments)
    end
    CUDA.synchronize()
    kstop = time_ns()
    _ = Array(d_state_flux)
    _ = Array(d_fine_flux)
    stop = time_ns()

    println("Simulation Complete.")
    border_print()
    center_print("RESULTS SUMMARY")
    border_print()
    kernel_s = (kstop - kstart) * 1.0e-9
    offload_s = (stop - start) * 1.0e-9
    tpi = (kstop - kstart) / Float64(cfg.repeat) / Float64(cfg.segments) / Float64(cfg.egroups)
    @printf("%-25s%.3f seconds\n", "Total kernel time:", kernel_s)
    @printf("%-25s%.3f seconds\n", "Device offload time:", offload_s)
    @printf("%-25s%.3f ns\n", "Time per Intersection:", tpi)
    border_print()
end

main()
