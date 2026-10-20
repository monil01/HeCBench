using CUDA, KernelAbstractions
KernelAbstractions.@kernel function source_zero_ka!(x,n)
    i=KernelAbstractions.@index(Global,Linear)
    if i<=n;x[i]=0f0;end
end
KernelAbstractions.@kernel function source_cbow_ka!(::Val{A},d,v,w,neg,ns,t,sen,table,s0,s1,rng,ex) where {A}
    lid=KernelAbstractions.@index(Local,Linear)-1
    pos=KernelAbstractions.@index(Group,Linear)-1
    a=A
    sh=KernelAbstractions.@localmem(Float32,(128+2*A,))
    @inbounds begin
# Source CBOW index body. Thread group128, one word position per block.
# KernelAbstractions.@synchronize() is expanded to the owning frontend's own group synchronization.
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
                    KernelAbstractions.@synchronize()
                    if lid < 64
                        sh[lid+1] += sh[lid+65]
                    end
                    KernelAbstractions.@synchronize()
                    for step in (32,16,8,4,2,1)
                        if lid < 32
                            sh[lid+1] += sh[lid+step+1]
                        end
                        KernelAbstractions.@synchronize()
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
                    KernelAbstractions.@synchronize()
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
        KernelAbstractions.@synchronize()
    end
    if lid == 0
        rng[pos+1] = seed
    end
end

    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch_zero!(x,n)
    source_zero_ka!(KernelAbstractions.get_backend(x),256)(x,n;ndrange=cld(n,256)*256)
end
function launch_cbow!(c)
    d,a,v,w,neg,ns,blocks,t,cold=c.h;s0,s1,sen,rng,table,ex=c.gpu
    source_cbow_ka!(KernelAbstractions.get_backend(s0),128)(Val(a),d,v,w,neg,ns,t,sen,table,s0,s1,rng,ex;ndrange=blocks*128)
end
main()
