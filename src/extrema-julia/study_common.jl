using CUDA,Printf
const ROOT=@__DIR__
const GUARD=UInt8(165)
function value_bits(a,ty)
 ty in (1,3) ? UInt64.(reinterpret(UInt32,a)) : collect(reinterpret(UInt64,a))
end
function study_main(model)
 fixture=collect(reinterpret(UInt64,read(joinpath(ROOT,"fixture.bin"))));gold=collect(reinterpret(UInt64,read(joinpath(ROOT,"oracle.bin"))))
 arrays=[];paths=[];pos=1
 for ai in 1:12
  ty,dim,nx,ny=Int.(fixture[pos:pos+3]);pos+=4;n=nx*ny;bits=fixture[pos:pos+n-1];pos+=n
  gb=ty in (1,3) ? UInt64(0xA5A5A5A5) : UInt64(0xA5A5A5A5A5A5A5A5)
  typedbits=ty in (1,3) ? UInt32.(vcat(fill(gb,8),bits,fill(gb,8))) : vcat(fill(gb,8),bits,fill(gb,8))
  T=(Int32,Int64,Float32,Float64)[ty];raw=collect(reinterpret(T,typedbits));w=CuArray(raw);a=@view w[9:end-8];r=CUDA.fill(GUARD,n+16);payload=@view r[9:end-8]
  push!(arrays,(ty,dim,nx,ny,w,a,r,payload))
  combos=nx==1 ? [(1,true),(1,false)] : vcat([(o,true) for o in [1,2,4,8,16,32,64,128]],[(o,false) for o in [1,8,16]])
  for axis in (dim==1 ? [1] : [0,1]),(o,c) in combos
   push!(paths,(ai,o,c,axis));launch_extrema(model,a,payload,dim,nx,ny,o,c,axis)
  end
 end
 CUDA.synchronize()
 for trial in 1:5
  prefix=UInt64[];out=UInt64[];times=Float64[];elapsed=0.0;counts=Int[]
  for (ty,dim,nx,ny,w,a,r,payload) in arrays
   append!(prefix,UInt64[ty,dim,nx,ny]);append!(prefix,value_bits(Array(a),ty))
  end
  for (ai,o,c,axis) in paths
   ty,dim,nx,ny,w,a,r,payload=arrays[ai];fill!(r,GUARD);CUDA.synchronize();stamp=time_ns()
   for rep in 1:100;launch_extrema(model,a,payload,dim,nx,ny,o,c,axis);end
   CUDA.synchronize();dt=(time_ns()-stamp)*1e-9/100;elapsed+=dt;push!(times,dt)
   rr=Array(r);push!(counts,count(==(UInt8(1)),rr[9:end-8]));append!(out,UInt64.(rr));append!(out,value_bits(Array(w),ty))
  end
  mismatches=sum(a!=b for (a,b) in zip(out,gold));ok=reinterpret(UInt8,prefix)==reinterpret(UInt8,fixture)&&mismatches==0
  words=vcat(prefix,out);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,words)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":null,\"max_rel_error\":null,\"artifact_word_mismatches\":%d,\"path_times_seconds\":[%s],\"path_true_counts\":[%s],\"timing_region\":\"sum148_hostinclusive_100launch_batches_div100\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(words),mismatches,join(string.(times),","),join(string.(counts),","),file,length(words))
 end
end
