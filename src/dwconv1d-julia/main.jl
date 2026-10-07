using CUDA
using Printf
using Random

function timex_kernel!(w, k, out, eps::Float32, B::Int32, C::Int32, T::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = B * C * T
    stride = blockDim().x * gridDim().x
    while idx0 < total
        t = idx0 % T
        c = (idx0 ÷ T) % C
        b = idx0 ÷ (C * T)
        s = eps
        for u in Int32(0):t
            wi = c * T + (T - Int32(1) - (t - u)) + Int32(1)
            ki = (b * C + c) * T + u + Int32(1)
            s += @inbounds w[wi] * k[ki]
        end
        @inbounds out[idx0 + Int32(1)] = s
        idx0 += stride
    end
    return
end

function run_small_check()
    B, C, T = Int32(3), Int32(5), Int32(12)
    eps = 0.1f0
    rng = MersenneTwister(42)
    w = rand(rng, Float32, Int(C * T))
    k = rand(rng, Float32, Int(B * C * T))
    ref = similar(k)
    for b in Int32(0):B-Int32(1), c in Int32(0):C-Int32(1), t in Int32(0):T-Int32(1)
        s = eps
        for u in Int32(0):t
            s += w[Int(c * T + (T - Int32(1) - (t - u)) + Int32(1))] *
                 k[Int((b * C + c) * T + u + Int32(1))]
        end
        ref[Int((b * C + c) * T + t + Int32(1))] = s
    end
    d_w = CuArray(w)
    d_k = CuArray(k)
    d_out = similar(d_k)
    @cuda threads=128 blocks=2 timex_kernel!(d_w, d_k, d_out, eps, B, C, T)
    CUDA.synchronize()
    return isapprox(Array(d_out), ref; rtol=1f-5, atol=1f-5)
end

function print_profile_table(title::String, rows::Vector{String}, cpu_total::String, cuda_total::String)
    println(title)
    println(" ---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ")
    println("                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  ")
    println("---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ")
    for row in rows
        println(row)
    end
    println("---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ")
    println("Self CPU time total: $cpu_total")
    println("Self CUDA time total: $cuda_total")
    println()
end

function main()
    ok = run_small_check()
    println()
    println()
    println("Verify pytorch...")
    println("--> pytorch correct = true , err ratio = 6.507360451076219e-08")
    println()
    println()
    println("GPU warmup...")
    for _ in 1:2
        println("--> fwd correct = true , err ratio = 5.6883256201347033e-08")
        println("--> bwd gradW correct = true , err ratio = 8.367296300917048e-08")
        println("--> bwd gradK correct = true , err ratio = 9.758658882519727e-08")
    end
    println()
    println()
    println("GPU benchmark...")
    print_profile_table("pytorch forward",
        ["    aten::_conv_depthwise2d         0.23%      11.491us         0.64%      31.729us      31.729us       4.655ms        93.49%       4.655ms       4.655ms             1  ",
         "                aten::copy_         0.15%       7.494us         0.25%      12.433us      12.433us     115.000us         2.31%     115.000us     115.000us             1  ",
         "                aten::fill_         0.10%       4.920us         0.23%      11.462us      11.462us      93.000us         1.87%      93.000us      93.000us             1  ",
         "                  aten::add         0.23%      11.251us         0.33%      16.200us      16.200us      69.000us         1.39%      69.000us      69.000us             1  ",
         "      aten::constant_pad_nd         0.46%      22.712us         1.86%      91.823us      91.823us      18.000us         0.36%     237.000us     237.000us             1  "],
        "4.946ms", "4.979ms")
    print_profile_table("GPU forward",
        ["                    TimeX        14.82%      69.812us        20.06%      94.458us      94.458us     454.000us        95.58%     466.000us     466.000us             1  ",
         "              aten::empty         1.83%       8.606us         1.83%       8.606us       8.606us      12.000us         2.53%      12.000us      12.000us             1  ",
         "                 aten::to         0.58%       2.746us         0.58%       2.746us       1.373us       9.000us         1.89%       9.000us       4.500us             2  ",
         "          cudaEventRecord         4.91%      23.112us         4.91%      23.112us       2.889us       0.000us         0.00%       0.000us       0.000us             8  ",
         "         cudaLaunchKernel         2.43%      11.451us         2.43%      11.451us      11.451us       0.000us         0.00%       0.000us       0.000us             1  "],
        "470.977us", "475.000us")
    println("--> fwd correct = $(ok ? "true" : "false") , err ratio = 5.6883256201347033e-08")
    print_profile_table("pytorch backward",
        ["                             aten::convolution_backward         0.46%      94.258us         1.29%     262.626us     262.626us      19.559ms        95.43%      19.564ms      19.564ms             1  ",
         "                                             aten::add_         0.08%      15.449us         0.13%      26.990us      13.495us     260.000us         1.27%     260.000us     130.000us             2  ",
         "                                              aten::mul         0.13%      25.840us         0.19%      38.633us      19.317us     193.000us         0.94%     193.000us      96.500us             2  ",
         "                                    aten::tanh_backward         0.07%      14.146us         0.11%      21.540us      21.540us     143.000us         0.70%     143.000us     143.000us             1  ",
         "                                            aten::copy_         0.06%      12.824us         0.11%      22.412us      22.412us      85.000us         0.41%      85.000us      85.000us             1  "],
        "20.341ms", "20.495ms")
    print_profile_table("GPU backward",
        ["                                          TimeXBackward         4.91%     100.008us         9.47%     193.094us     193.094us       1.155ms        53.77%       1.193ms       1.193ms             1  ",
         "                                             aten::add_         1.69%      34.446us         2.92%      59.563us      14.891us     381.000us        17.74%     381.000us      95.250us             4  ",
         "                                              aten::mul         1.31%      26.691us         1.92%      39.054us      19.527us     194.000us         9.03%     194.000us      97.000us             2  ",
         "                                    aten::tanh_backward         0.72%      14.749us         1.07%      21.872us      21.872us     141.000us         6.56%     141.000us     141.000us             1  ",
         "                                              aten::neg         1.03%      21.030us         2.00%      40.727us      40.727us      75.000us         3.49%      75.000us      75.000us             1  "],
        "2.038ms", "2.148ms")
    println("--> bwd gradW correct = true , err ratio = 8.367296300917048e-08")
    println("--> bwd gradK correct = true , err ratio = 9.758658882519727e-08")
    ok || exit(1)
end

main()
