function compute_permutations()
    permutations = Vector{UInt32}(undef, 1024)
    indices = zeros(Int32, 16)
    num = 1
    imax = 15
    for i in imax:-1:0
        for m in i:15
            indices[m+1] = 2
        end
        jmax = i == 0 ? 15 : 16
        for j in jmax:-1:i
            if j < 16
                indices[j+1] = 1
            end
            permutation = UInt32(0)
            for p in 0:15
                permutation |= UInt32(indices[p+1]) << UInt32(2p)
            end
            permutations[num] = permutation
            num += 1
        end
    end
    for _ in 1:9
        permutations[num] = 0x000aa555
        num += 1
    end
    fill!(indices, 0)
    for i in imax:-1:0
        for m in i:15
            indices[m+1] = 2
        end
        jmax = i == 0 ? 15 : 16
        for j in jmax:-1:i
            for m in j:15
                indices[m+1] = 3
            end
            kmax = j == 0 ? 15 : 16
            for k in kmax:-1:j
                if k < 16
                    indices[k+1] = 1
                end
                permutation = UInt32(0)
                has_three = false
                for p in 0:15
                    permutation |= UInt32(indices[p+1]) << UInt32(2p)
                    has_three |= indices[p+1] == 3
                end
                if has_three
                    permutations[num] = permutation
                    num += 1
                end
            end
        end
    end
    for _ in 1:49
        permutations[num] = 0x00aaff55
        num += 1
    end
    return permutations
end

