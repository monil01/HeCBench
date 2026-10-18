const INV32=2.3283064f-10
const HALF32=1.1641532f-10
@inline function philox4(counter,seed)
 c=counter;k0=seed%UInt32;k1=(seed>>32)%UInt32
 @inbounds for round in 1:10
  a=UInt64(0xd2511f53)*UInt64(c[1]);b=UInt64(0xcd9e8d57)*UInt64(c[3])
  c=((b>>32)%UInt32 ⊻ c[2] ⊻ k0,b%UInt32,(a>>32)%UInt32 ⊻ c[4] ⊻ k1,a%UInt32)
  k0+=UInt32(0x9e3779b9);k1+=UInt32(0xbb67ae85)
 end
 return c
end
@inline function increment_counter(c)
 a=c[1]+UInt32(1);b=c[2]+UInt32(a==0);cc=c[3]+UInt32(a==0&&b==0);d=c[4]+UInt32(a==0&&b==0&&cc==0)
 return a,b,cc,d
end
@inline function initial_rng(seed,subsequence,offset)
 base=offset>>2;c=(base%UInt32,(base>>32)%UInt32,subsequence%UInt32,(subsequence>>32)%UInt32)
 return c,philox4(c,seed)
end
@inline function rng_next(counter,cache,seed,offset)
 counter=increment_counter(counter);next=philox4(counter,seed);r=Int32(offset&UInt64(3))
 raw=ntuple(i->(@inbounds i+r<=4 ? cache[i+r] : next[i+r-4]),Val(4))
 return counter,next,raw
end
@inline rng_uniform(x)=Float32(x)*INV32+HALF32
