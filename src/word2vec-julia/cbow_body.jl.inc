# Source CBOW index body. Thread group128, one word position per block.
# BARRIER is expanded to the owning frontend's own group synchronization.
# Extra full-group synchronization stages explicitly implement the source
# volatile warp-lockstep reduction; this resource/ordering adaptation is disclosed.
if pos < 1024
    seed = rng[pos+1]
    for si in 0:ns-1
        for c in lid:128:d-1
            sh[128+c+1] = 0f0
            sh[128+a+c+1] = 0f0
        end
        seed = seed * UInt32(1664525) + UInt32(1013904223)
        b = Int(rem(seed,UInt32(w)))
        word = Int(sen[si*1024+pos+1])
        cw = 0
        for z in b:2*w-b
            if z != w
                near = pos-w+z
                if 0 <= near < 1024
                    lastword = Int(sen[si*1024+near+1])
                    for c in lid:128:d-1
                        sh[128+c+1] += s0[lastword*a+c+1]
                    end
                    cw += 1
                end
            end
        end
        if cw > 0
            for c in lid:128:d-1
                sh[128+c+1] /= Float32(cw)
            end
            alpha = reinterpret(Float32,sen[6144+si+1])
            if neg > 0
                for draw in 0:neg
                    if draw == 0
                        target = word
                        label = 1f0
                    else
                        seed = seed * UInt32(1664525) + UInt32(1013904223)
                        target = Int(table[Int(rem(seed,UInt32(t)))+1])
                        if target == 0
                            target = Int(rem(seed,UInt32(v-1)))+1
                        end
                        if target == word
                            continue
                        end
                        label = 0f0
                    end
                    sh[lid+1] = 0f0
                    for c in lid:128:d-1
                        sh[lid+1] += sh[128+c+1] * s1[target*a+c+1]
                    end
                    BARRIER
                    if lid < 64
                        sh[lid+1] += sh[lid+65]
                    end
                    BARRIER
                    for step in (32,16,8,4,2,1)
                        if lid < 32
                            sh[lid+1] += sh[lid+step+1]
                        end
                        BARRIER
                    end
                    dot = sh[1]
                    if dot > 6f0
                        g = (label-1f0)*alpha
                    elseif dot < -6f0
                        g = label*alpha
                    else
                        ei = trunc(Int,(dot+6f0)*83f0)
                        g = (label-ex[ei+1])*alpha
                    end
                    for c in lid:128:d-1
                        sh[128+a+c+1] += g*s1[target*a+c+1]
                    end
                    for c in lid:128:d-1
                        s1[target*a+c+1] += g*sh[128+c+1]
                    end
                    BARRIER
                end
            end
            for z in b:2*w-b
                if z != w
                    near = pos-w+z
                    if 0 <= near < 1024
                        lastword = Int(sen[si*1024+near+1])
                        for c in lid:128:d-1
                            s0[lastword*a+c+1] += sh[128+a+c+1]
                        end
                    end
                end
            end
        end
        BARRIER
    end
    if lid == 0
        rng[pos+1] = seed
    end
end
