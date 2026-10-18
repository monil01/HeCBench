using CUDA,Printf
const ROOT=@__DIR__
include(joinpath(ROOT,"permutations_reference.jl"))
const GUARD=UInt32(0xdeadbeef)
function block_mse(words,image,b)
 w=words[2b+1];inds=words[2b+2];a=w&0xffff;c=w>>16
 expand(x)=((Int((x>>11)&31)*255÷31),(Int((x>>5)&63)*255÷63),(Int(x&31)*255÷31))
 # BC1 integerbit expansion, not division255 approximation.
 function rgb(x)
  r=Int((x>>11)&31);g=Int((x>>5)&63);b=Int(x&31)
  return ((r<<3)|(r>>2),(g<<2)|(g>>4),(b<<3)|(b>>2))
 end
 p0=rgb(a);p1=rgb(c)
 p2=a>c ? ntuple(k->(2p0[k]+p1[k])÷3,3) : ntuple(k->(p0[k]+p1[k])÷2,3)
 p3=a>c ? ntuple(k->(p0[k]+2p1[k])÷3,3) : (0,0,0)
 palette=(p0,p1,p2,p3);err=0
 for j in 0:15
  col=palette[Int((inds>>(2j))&3)+1];x=image[16b+j+1]
  for k in 1:3;diff=col[k]-Int((x>>(8(k-1)))&255);err+=diff*diff;end
 end
 err/48
end
function study_main(model)
 fixture=collect(reinterpret(UInt32,read(joinpath(ROOT,"fixture.bin"))))
 nb=Int(fixture[1]);perms=fixture[6:1029];image=fixture[1030:end]
 @assert compute_permutations()==perms
 pw=CuArray(vcat(fill(GUARD,8),perms,fill(GUARD,8)));iw=CuArray(vcat(fill(GUARD,8),image,fill(GUARD,8)));rw=CUDA.fill(GUARD,nb*2+16)
 p=@view pw[9:end-8];i=@view iw[9:end-8];r=@view rw[9:end-8];template=CUDA.zeros(Float32,467)
 for (n,off) in [(12,0),(5,2)];launch_compress(model,p,i,r,template,n,Int32(off));end
 CUDA.synchronize()
 for trial in 1:5
  got=UInt32[];prefix=UInt32[];dt=0.0;times=Float64[];quality=Float64[];ok=true
  for (path,(n,off)) in enumerate([(12,0),(5,2)])
   fill!(rw,GUARD);CUDA.synchronize();t=time_ns()
   for repeat in 1:10;launch_compress(model,p,i,r,template,n,Int32(off));end
   CUDA.synchronize();secs=(time_ns()-t)*1e-9/10;dt+=secs;push!(times,secs)
   pp=Array(pw);ii=Array(iw);rr=Array(rw);payload=rr[9:end-8]
   ok &= pp==vcat(fill(GUARD,8),perms,fill(GUARD,8)) && ii==vcat(fill(GUARD,8),image,fill(GUARD,8))
   ok &= rr[1:8]==fill(GUARD,8)&&rr[end-7:end]==fill(GUARD,8)
   for b in 0:nb-1
    if off<=b<off+n
     mse=block_mse(payload,image,b);push!(quality,mse);ok &= isfinite(mse)&&mse<=256
    else;ok &= payload[2b+1]==GUARD&&payload[2b+2]==GUARD;end
   end
   if path==1;append!(prefix,fixture[1:5]);append!(prefix,pp[9:end-8]);append!(prefix,ii[9:end-8]);end
   append!(got,rr);append!(got,pp);append!(got,ii)
  end
  ok &= prefix==fixture
  out=vcat(prefix,got);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"max_fixture_RGB_MSE\":%.17g,\"path_times_seconds\":[%s],\"timing_region\":\"sum2_hostinclusive_10kernel_batches_div10\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,dt,string(ok),length(out),maximum(quality),join(string.(times),","),file,length(out))
 end
end
