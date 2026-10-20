using CUDA
include(joinpath(@__DIR__,"..","device_primitives.jl"))
function match_kernel(input,fs,ds,tf,td)
 tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid
 s=CUDA.@cuStaticSharedMem(UInt8,5188)
 @inbounds begin
for k in 0:3
    j=tid+128*k
    storeword!(s,4*j,input[gid*512+j+1])
end
CUDA.sync_threads()
for k in 0:3
    j=tid+128*k;look=j;win=max(j-32,0);best=0;bestoff=0;len=0;off=0
    while win<j && look<512
        if loadword(s,4*look)==loadword(s,4*win)
            if off==0;off=look-win;end
            len+=1;look+=1
        else
            if len>best;best=len;bestoff=off;end
            len=0;off=0;look=j
        end
        win+=1
    end
    if len>best;best=len;bestoff=off;end
    s[2048+j+1]=UInt8(best);s[2560+j+1]=UInt8(bestoff)
    storeword!(s,3072+4*j,UInt32(0))
end
CUDA.sync_threads()
if tid==0
    flagpos=UInt8(1);flag=UInt8(0);count=0;j=0
    while j<512
        len=Int(s[2048+j+1])
        if len<1
            storeword!(s,3072+4*j,UInt32(4));j+=1
        else
            storeword!(s,3072+4*j,UInt32(2));j+=len;flag|=flagpos
        end
        if flagpos==0x80
            s[5124+count+1]=flag;count+=1;flagpos=UInt8(1);flag=UInt8(0)
        else
            flagpos<<=1
        end
    end
    if flagpos!=1;s[5124+count+1]=flag;count+=1;end
    fs[gid+1]=UInt32(count)
end
CUDA.sync_threads()
offset=1;d=256
while d>0
    for k in 0:3
        j=tid+128*k
        if j<d
            ai=offset*(2*j+1)-1;bi=offset*(2*j+2)-1
            storeword!(s,3072+4*bi,loadword(s,3072+4*bi)+loadword(s,3072+4*ai))
        end
        CUDA.sync_threads()
    end
    offset*=2;d>>=1
end
if tid==0
    total=loadword(s,3072+4*511);ds[gid+1]=total
    storeword!(s,3072+4*512,total);storeword!(s,3072+4*511,UInt32(0))
end
CUDA.sync_threads()
d=1
while d<512
    offset>>=1
    for k in 0:3
        j=tid+128*k
        if j<d
            ai=offset*(2*j+1)-1;bi=offset*(2*j+2)-1
            a=loadword(s,3072+4*ai);b=loadword(s,3072+4*bi)
            storeword!(s,3072+4*ai,b);storeword!(s,3072+4*bi,a+b)
        end
        CUDA.sync_threads()
    end
    d*=2
end
for k in 0:3
    j=tid+128*k;a=Int(loadword(s,3072+4*j));b=Int(loadword(s,3072+4*(j+1)))
    if a!=b
        if s[2048+j+1]<1
            for z in 0:3;td[gid*2048+a+z+1]=s[4*j+z+1];end
        else
            td[gid*2048+a+1]=s[2048+j+1];td[gid*2048+a+2]=s[2560+j+1]
        end
    end
end
if tid==0
    for j in 0:Int(fs[gid+1])-1;tf[gid*64+j+1]=s[5124+j+1];end
end

end
return nothing
end
function scan_kernel(input,out,sums,n)
 tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid
 s=CUDA.@cuStaticSharedMem(UInt8,2048)
 @inbounds begin
for k in 0:3
    j=tid+128*k;globalj=gid*512+j
    v=globalj<n ? input[globalj+1] : UInt32(0)
    storeword!(s,4*j,v)
end
CUDA.sync_threads()
offset=1;d=256
while d>0
    for k in 0:3
        j=tid+128*k
        if j<d
            ai=offset*(2*j+1)-1;bi=offset*(2*j+2)-1
            storeword!(s,4*bi,loadword(s,4*bi)+loadword(s,4*ai))
        end
    end
    CUDA.sync_threads()
    offset*=2;d>>=1
end
if tid==0
    sums[gid+1]=loadword(s,4*511);storeword!(s,4*511,UInt32(0))
end
CUDA.sync_threads()
d=1
while d<512
    offset>>=1
    for k in 0:3
        j=tid+128*k
        if j<d
            ai=offset*(2*j+1)-1;bi=offset*(2*j+2)-1
            a=loadword(s,4*ai);b=loadword(s,4*bi)
            storeword!(s,4*ai,b);storeword!(s,4*bi,a+b)
        end
    end
    CUDA.sync_threads()
    d*=2
end
for k in 0:3
    j=tid+128*k;globalj=gid*512+j
    if globalj<n;out[globalj+1]=loadword(s,4*j);end
end

end
return nothing
end
function pack_kernel(fo,doo,tf,td,pf,pd)
 tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid
 @inbounds begin
j=tid
while j<Int(fo[gid+2]-fo[gid+1])
    pf[Int(fo[gid+1])+j+1]=tf[gid*64+j+1];j+=128
end
j=tid
while j<Int(doo[gid+2]-doo[gid+1])
    pd[Int(doo[gid+1])+j+1]=td[gid*2048+j+1];j+=128
end

end
return nothing
end
function decode_kernel(out,fo,doo,pf,pd,nb)
 tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid
 @inbounds begin
if i<nb
    at=Int(doo[i+1]);pos=0;token=0
    while pos<512
        flag=(pf[Int(fo[i+1])+div(token,8)+1]>>(token%8))&1;token+=1
        if flag==1
            len=Int(pd[at+1]);off=Int(pd[at+2]);at+=2
            for z in 1:len
                out[i*512+pos+1]=out[i*512+pos-off+1];pos+=1
            end
        else
            out[i*512+pos+1]=loadword(pd,at);at+=4;pos+=1
        end
    end
end

end
return nothing
end
function add_kernel(out,offsets,n)
 i=(Int(blockIdx().x)-1)*Int(blockDim().x)+Int(threadIdx().x)-1
 if i<n;@inbounds out[i+1]+=offsets[div(i,512)+1];end
 return nothing
end
alloc_backend(::Type{T},n) where T=CUDA.CuArray{T}(undef,n)
function run_match!(input,fs,ds,tf,td,nb);CUDA.@cuda threads=128 blocks=nb match_kernel(input,fs,ds,tf,td);end
function run_scan!(input,out,sums,n,groups);CUDA.@cuda threads=128 blocks=groups scan_kernel(input,out,sums,n);end
function run_add!(out,offsets,n);CUDA.@cuda threads=128 blocks=cld(n,128) add_kernel(out,offsets,n);end
function run_pack!(fo,doo,tf,td,pf,pd,nb);CUDA.@cuda threads=128 blocks=nb pack_kernel(fo,doo,tf,td,pf,pd);end
function run_decode!(out,fo,doo,pf,pd,nb);CUDA.@cuda threads=32 blocks=cld(nb,32) decode_kernel(out,fo,doo,pf,pd,nb);end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
