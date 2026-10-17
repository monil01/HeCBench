using CUDA,Printf
const COEFFS=((1f0,-9f0,28f0,-33f0),(1f0,1f0,1f0,1f0),(1f0,0f0,-2f0,0f0),(2f0,2f0,2f0,2f0),(1f0,0f0,2f0,1f0),(1f0,0f0,-4f0,1f0),(1f0,0f0,3f0,-2f0),(2f0,0f0,-8f0,2f0))
function fixture()
 n=1027;inputs=[CUDA.pin(zeros(Float32,n)) for _ in 1:5]
 for i in 1:n
  z=COEFFS[(i-1)%length(COEFFS)+1]
  for k in 1:4;inputs[k][i]=z[k];end
  inputs[5][i]=Float32((i-1)%17-8)*0.25f0
 end
 inputs
end
# Independent Float64 reconstruction of the SOURCE root-selection behavior, not minimizer accuracy.
function behavior_oracle(A,B,C,D)
 b=.75*Float64(B)/A;c=.5*Float64(C)/A;d=.25*Float64(D)/A
 q=round((c/3-b*b/9)*1e5,RoundNearestTiesAway)/1e5
 rr=round((b*c/6-b*b*b/27-.5d)*1e5,RoundNearestTiesAway)/1e5
 disc=rr*rr+q*q*q
 if disc<=1e-5
  th=acos(rr/sqrt(-q*q*q));v=sort([2sqrt(-q)*cos((th+k*2Float64(Float32(3.1415927)))/3)-b/3 for k in 0:2]);lo,hi=v[1],v[3]
  delta=A*(hi^4-lo^4)/4+B*(hi^3-lo^3)/3+C*(hi^2-lo^2)/2+D*(hi-lo)
  delta<=0 ? hi : lo
 else
  cbrt(rr+sqrt(disc))+cbrt(rr-sqrt(disc))-b/3
 end
end
function qem_call!(launch!,inputs,out,streamed)
 n=length(out);streams=streamed ? [CUDA.CuStream() for _ in 1:5] : CUDA.CuStream[]
 a=ntuple(_->CuArray{Float32}(undef,n),20) # preserve original unused E allocation too
 GC.@preserve a inputs out begin
  if streamed
   size=n÷4
   for k in 0:4
    len=k==4 ? n%4 : size;len==0&&continue;offset=k*size;s=streams[k+1]
    CUDA.stream!(s) do
     for j in 1:4;unsafe_copyto!(pointer(a[j],offset+1),pointer(inputs[j],offset+1),len;async=true,stream=s);end
     av=map(z->view(z,offset+1:offset+len),a);launch!(av,Int32(len))
     unsafe_copyto!(pointer(out,offset+1),pointer(a[19],offset+1),len;async=true,stream=s)
    end
   end
   CUDA.synchronize()
  else
   for j in 1:4;copyto!(a[j],inputs[j]);end
   launch!(a,Int32(n));copyto!(out,a[19]);CUDA.synchronize()
  end
 end
 foreach(CUDA.unsafe_free!,a)
 foreach(s->finalize(s),streams) # run registered CUDA stream destructor once
 out
end
function study_main(launch!)
 rep=isempty(ARGS) ? 3 : parse(Int,ARGS[1]);inputs=fixture();n=length(inputs[1]);outs=[CUDA.pin(zeros(Float32,n)) for _ in 1:2];gold=[behavior_oracle((inputs[j][i] for j in 1:4)...) for i in 1:n];prefix=vcat(Float32[n],inputs...)
 for mode in 1:2;qem_call!(launch!,inputs,outs[mode],mode==2);end
 for trial in 1:5
  CUDA.synchronize();gb=Base.gc_num();st=time_ns()
  for _ in 1:rep,mode in 1:2;qem_call!(launch!,inputs,outs[mode],mode==2);end
  CUDA.synchronize();elapsed=(time_ns()-st)*1e-9/rep;ga=Base.gc_num();got=vcat(outs...);ma=maximum(abs(Float64(outs[m][i])-gold[i]) for m in 1:2,i in 1:n);ok=all(isfinite,got)&&ma<=1e-3
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum_serial_and_four_plus_remainder_streams_fullhost_20alloc_H2D_two_stages_D2H_free_stream_lifecycle\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),ma,rep,(ga.total_time-gb.total_time)*1e-9,ga.pause-gb.pause,path,length(art));ok||error("finite sourcebehavior oracle; NOT global quartic minimizer")
 end
end
