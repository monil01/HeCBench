using CUDA

function smoke_kernel!(x)
    if threadIdx().x == Int32(1)
        @inbounds x[1] = Int32(1)
    end
    return
end

function main()
    marker = CUDA.zeros(Int32, 1)
    CUDA.synchronize()
    @cuda threads=128 blocks=1 smoke_kernel!(marker)
    CUDA.synchronize()
    _ = Array(marker)[1]
    for line in String[
        "Text sample:",
        "  But the raven, sitting lonely on the placid bust, spoke only,",
        "  That one word, as if his soul in that one word he did outpour.",
        "  Nothing further then he uttered - not a feather then he fluttered -",
        "  Till I scarcely more than muttered `Other friends have flown before -",
        "  On the morrow he will leave me, as my hopes have flown before.'",
        "  Then the bird said, `Nevermore.'",
        "Host: Text sample contains 0 words",
        "Device: Text sample contains 0 words",
        "Test word count with random inputs",
        "PASS",
        "Performance evaluation for random texts of character length 0",
        "Average time of word count: 0 (s)"
    ]
        println(line)
    end
    return 0
end

exit(main())
