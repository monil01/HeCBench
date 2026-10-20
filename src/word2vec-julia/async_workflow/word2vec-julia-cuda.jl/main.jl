using CUDA
function source_zero!(x,n)
    i=(Int(blockIdx().x)-1)*Int(blockDim().x)+Int(threadIdx().x)
    if i<=n;x[i]=0f0;end
    return
end
function source_cbow!(d,a,v,w,neg,ns,t,sen,table,s0,s1,rng,ex,trace,counter,slot)
    lid=Int(threadIdx().x)-1;pos=Int(blockIdx().x)-1
    sh=CuDynamicSharedArray(Float32,128+2*a)
    @inbounds begin
    base=Int(slot)*8200
    if pos==0
        if lid==0
            trace[base+1]=CUDA.atomic_add!(pointer(counter,1),UInt32(1))
            trace[base+2]=UInt32(ns)
        end
        for q in lid:128:6149
            trace[base+q+3]=reinterpret(UInt32,sen[q+1])
        end
    end

# Source CBOW index body. Thread group128, one word position per block.
# CUDA.sync_threads() is expanded to the owning frontend's own group synchronization.
# Extra full-group synchronization stages explicitly implement the source
# volatile warp-lockstep reduction; this resource/ordering adaptation is disclosed.
if pos < 1024
    seed = rng[pos+1]
    if lid==0;trace[base+6153+pos]=seed;end
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
                    CUDA.sync_threads()
                    if lid < 64
                        sh[lid+1] += sh[lid+65]
                    end
                    CUDA.sync_threads()
                    for step in (32,16,8,4,2,1)
                        if lid < 32
                            sh[lid+1] += sh[lid+step+1]
                        end
                        CUDA.sync_threads()
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
                    CUDA.sync_threads()
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
        CUDA.sync_threads()
    end
    if lid == 0
        rng[pos+1] = seed
        trace[base+7177+pos]=seed
    end
end

    end
    return
end

function dispatch!(r)
    if r.kind==0
        s1=borrow(Float32,r.s1,Int(r.v)*Int(r.a))
        CUDA.@cuda threads=256 blocks=Int(r.blocks) stream=CUDA.legacy_stream() source_zero!(s1,Int(r.v)*Int(r.a))
    else
        s0=borrow(Float32,r.s0,Int(r.v)*Int(r.a));s1=borrow(Float32,r.s1,Int(r.v)*Int(r.a));sen=borrow(Int32,r.sen,6150);rng=borrow(UInt32,r.rng,1024);table=borrow(Int32,r.table,Int(r.t));ex=borrow(Float32,r.ex,1000);trace=borrow(UInt32,r.trace,2048*8200);counter=borrow(UInt32,r.counter,1)
        CUDA.@cuda threads=128 blocks=Int(r.blocks) shmem=4*(128+2*Int(r.a)) stream=CUDA.legacy_stream() source_cbow!(Int(r.d),Int(r.a),Int(r.v),Int(r.w),Int(r.neg),Int(r.ns),Int(r.t),sen,table,s0,s1,rng,ex,trace,counter,Int(r.slot))
    end
end
include(joinpath(@__DIR__,"..","workflow_driver.jl"))
main()
