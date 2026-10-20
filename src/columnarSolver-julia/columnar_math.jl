# Fixed supplied educational fixture only. Exact source algorithms and draw order.
const CS_SA=CUDA.CUDACore.CUDAKernels.StaticArrays
@inline function cs_step(s::UInt32)
    (UInt32(26757677)*s+UInt32(1))&UInt32(0x7fffffff)
end
@inline function cs_random(s::UInt32)
    next=cs_step(s)
    Float32(next)/2147483648f0,next
end
@inline cs_index(r::Float32)=min(trunc(Int32,r*30f0),Int32(29))
@inline function cs_swap!(key,left::Int32,right::Int32)
    @inbounds if left!=right
        key[left+1]-=key[right+1]
        key[right+1]+=key[left+1]
        key[left+1]=key[right+1]-key[left+1]
    end
    nothing
end
@inline function cs_decrypt!(encrypted,key,decrypted,columns,lengths)
    @inbounds begin
        for j in Int32(0):Int32(29);lengths[j+1]=Int32(13)+(j<Int32(7));end
        for kp in Int32(0):Int32(29)
            off=Int32(0)
            for i in Int32(0):Int32(29)
                if key[i+1]<key[kp+1];off+=lengths[i+1];end
            end
            for j in Int32(0):lengths[kp+1]-Int32(1)
                columns[key[kp+1]*Int32(14)+j+1]=encrypted[off+j+1]
            end
        end
        for j in Int32(0):Int32(396)
            decrypted[j+1]=columns[key[rem(j,Int32(30))+1]*Int32(14)+div(j,Int32(30))+1]
        end
    end
    nothing
end
@inline function cs_decode_site!(idx,scores,encrypted,seeds,output,diag)
    key=CS_SA.MVector{30,Int32}(undef);backup=CS_SA.MVector{30,Int32}(undef);helper=CS_SA.MVector{30,Int32}(undef)
    localdec=CS_SA.MVector{397,Int32}(undef);bestdec=CS_SA.MVector{397,Int32}(undef)
    columns=CS_SA.MVector{420,Int32}(undef);lengths=CS_SA.MVector{30,Int32}(undef)
    @inbounds begin
        state=seeds[idx+1];draws=UInt32(0);bestscore=999999f0
        # Explicit private best-buffer initialization is a mapping covariate;
        # source first candidate always accepts under finite fixed positive scores.
        for j in Int32(0):Int32(396);bestdec[j+1]=Int32(0);end
        for j in Int32(0):Int32(29);key[j+1]=j;end
        for j in Int32(0):Int32(29)
            r,state=cs_random(state);draws+=UInt32(1);cs_swap!(key,j,cs_index(r))
        end
        for q in Int32(0):Int32(29);diag[idx*Int32(63)+q+1]=UInt32(key[q+1]);end
        for cycle in Int32(0):Int32(149999)
            for j in Int32(0):Int32(29);backup[j+1]=key[j+1];end
            temp=0f0
            r,state=cs_random(state);draws+=UInt32(1);branch=trunc(Int32,r*100f0)
            if branch<Int32(50)
                j=Int32(0)
                while true
                    r,state=cs_random(state);draws+=UInt32(1)
                    if !(Float32(j)<1f0+r*4f0);break;end
                    r,state=cs_random(state);draws+=UInt32(1);left=cs_index(r)
                    r,state=cs_random(state);draws+=UInt32(1);right=cs_index(r)
                    cs_swap!(key,left,right);j+=Int32(1)
                end
            elseif branch<Int32(70)
                j=Int32(0)
                while true
                    r,state=cs_random(state);draws+=UInt32(1)
                    if !(Float32(j)<1f0+r*2f0);break;end
                    r,state=cs_random(state);draws+=UInt32(1);bs=cs_index(r)
                    r,state=cs_random(state);draws+=UInt32(1);be=cs_index(r)
                    r,state=cs_random(state);draws+=UInt32(1);len=trunc(Int32,1f0+r*Float32(abs(bs-be)-Int32(1)))
                    for z in Int32(0):len-Int32(1)
                        cs_swap!(key,rem(bs+z,Int32(30)),rem(be+z,Int32(30)))
                    end
                    j+=Int32(1)
                end
            else
                r,state=cs_random(state);draws+=UInt32(1);l=trunc(Int32,1f0+r*28f0)
                r,state=cs_random(state);draws+=UInt32(1);f=trunc(Int32,r*29f0)
                r,state=cs_random(state);draws+=UInt32(1);t=rem(trunc(Int32,Float32(f+Int32(1))+r*28f0),Int32(30))
                for j in Int32(0):Int32(29);helper[j+1]=key[j+1];end
                t0=rem(t-f+Int32(30),Int32(30));n=rem(t0+l,Int32(30))
                for j in Int32(0):n-Int32(1)
                    ff=rem(f+j,Int32(30));tt=rem(rem(t0+j,n)+f,Int32(30));key[tt+1]=helper[ff+1]
                end
            end
            cs_decrypt!(encrypted,key,localdec,columns,lengths)
            for j in Int32(0):Int32(395)
                temp+=scores[Int32(26)*localdec[j+1]+localdec[j+2]+1]
            end
            if temp<bestscore
                bestscore=temp
                for j in Int32(0):Int32(396);bestdec[j+1]=localdec[j+1];end
            else
                for j in Int32(0):Int32(29);key[j+1]=backup[j+1];end
            end
        end
        for j in Int32(0):Int32(396);output[idx*Int32(397)+j+1]=bestdec[j+1];end
        for q in Int32(0):Int32(29);diag[idx*Int32(63)+Int32(30)+q+1]=UInt32(key[q+1]);end
        diag[idx*Int32(63)+61]=reinterpret(UInt32,bestscore);diag[idx*Int32(63)+62]=state;diag[idx*Int32(63)+63]=draws
    end
    nothing
end
