using CUDA
const P1=55;const P2=119;const P3=179;const P4=256
function first_column!(x,s,workspace)
 t=Int(threadIdx().x)-1;cx=CuStaticSharedArray(UInt32,512)
 cx[t+1]=x[t+1]
 CUDA.sync_threads()
 for k in 1:div(s,256)-1
  for i in 0:32:255
   if t<32
    j=256+i+t;cx[j+1]=cx[j-55+1]+cx[j-119+1]+cx[j-179+1]+cx[j-256+1]
   end
   CUDA.sync_threads()
  end
  cx[t+1]=cx[256+t+1];x[k*256+t+1]=cx[t+1]
  CUDA.sync_threads()
 end
 return nothing
end
function column_y!(y,s,workspace)
 t=Int(threadIdx().x)-1;cy=CuStaticSharedArray(UInt32,768)
 # Native initially reads undefined lower half, dead after >=2rounds; initialize for definedportableexecution.
 cy[t+1]=0;cy[256+t+1]=0;cy[512+t+1]=y[512+t+1]
 CUDA.sync_threads()
 for k in 0:div(s,256)-1
  cy[t+1]=cy[256+t+1];cy[256+t+1]=cy[512+t+1]
  CUDA.sync_threads()
  for i in 0:32:255
   if t<32
    j=512+i+t;cy[j+1]=cy[j-55+1]+cy[j-119+1]+cy[j-179+1]+cy[j-256+1]
   end
   CUDA.sync_threads()
  end
 end
 y[t+1]=cy[t+1];y[256+t+1]=cy[256+t+1];y[512+t+1]=cy[512+t+1]
 return nothing
end
function last_entries!(x,y,s,r,workspace)
 t=Int(threadIdx().x)-1;mem=CuStaticSharedArray(UInt32,2304)
 # a0[0:767], b0[768:1279], c0[1280:1791], d0[1792:2303]
 mem[t+1]=y[t+1]
 CUDA.sync_threads()
 if t<256;mem[t+512+1]=y[t+512+1];end
 CUDA.sync_threads()
 mem[1792+t+1]=mem[256+t+1];mem[1280+t+1]=mem[256+t+1];mem[768+t+1]=mem[256+t+1]
 CUDA.sync_threads()
 mem[768+t+1]+=mem[256-77+t+1]
 CUDA.sync_threads()
 mem[1280+t+1]+=mem[256-60+t+1]+mem[256-137+t+1]
 CUDA.sync_threads()
 mem[1792+t+1]+=mem[256-64+t+1]+mem[256-124+t+1]+mem[256-201+t+1]
 CUDA.sync_threads()
 for column in 1:r-1
  tmp=UInt32(0)
  if t<256
   for k in 0:76;tmp+=x[column*s-256+k+1]*mem[512+t-k+1];end
   for k in 0:59;tmp+=x[column*s-179+k+1]*mem[1024+t-k+1];end
   for k in 0:63;tmp+=x[column*s-119+k+1]*mem[1536+t-k+1];end
   for k in 0:54;tmp+=x[column*s-55+k+1]*mem[2048+t-k+1];end
   x[column*s+s-256+t+1]=tmp
  end
  CUDA.sync_threads()
 end
 return nothing
end
function columns!(x,s,r,workspace)
 t=Int(threadIdx().x)-1;block=Int(blockIdx().x)-1;warp=div(t,32);lane=t%32;cx=CuStaticSharedArray(UInt32,4096)
 # Initialize dead first-column workspace; native leaves it undefined but never exports that row.
 for i in 0:7;cx[i*512+t+1]=0;cx[i*512+256+t+1]=0;end
 first=block==0 ? 1 : 0;last=(block==cld(r,8)-1&&r%8!=0) ? r%8 : 8
 for i in first:last-1;cx[i*512+t+1]=x[(block*8+i)*s-256+t+1];end
 CUDA.sync_threads()
 for k in 0:div(s,256)-2
  for i in 0:32:255
   if !(block==0&&t==0)&&!(block==cld(r,8)-1&&warp>=last)
    j=warp*512+256+i+lane;cx[j+1]=cx[j-55+1]+cx[j-119+1]+cx[j-179+1]+cx[j-256+1]
   end
   CUDA.sync_threads()
  end
  for i in first:last-1
   cx[i*512+t+1]=cx[i*512+256+t+1];x[(block*8+i)*s+k*256+t+1]=cx[i*512+t+1]
  end
  CUDA.sync_threads()
 end
 return nothing
end
function pipeline!(x,y,s,r,streams,workspaces)
 dx=CUDA.cudaconvert(x);dy=CUDA.cudaconvert(y)
 GC.@preserve x y begin
  CUDA.stream!(streams[1]) do
   @cuda threads=256 blocks=1 first_column!(dx,s,nothing)
  end
  CUDA.stream!(streams[2]) do
   @cuda threads=256 blocks=1 column_y!(dy,s,nothing)
  end
  CUDA.synchronize(streams[1]);CUDA.synchronize(streams[2])
  @cuda threads=512 blocks=1 last_entries!(dx,dy,s,r,nothing)
  @cuda threads=256 blocks=cld(r,8) columns!(dx,s,r,nothing)
  CUDA.synchronize()
 end
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 16384;r=16;s=div(n,r);s%256==0&&s>=512||error("fixture shape")
 seed=UInt32[UInt32(i*1664525+1013904223) for i in 0:255];expected=zeros(UInt32,n);expected[1:256]=seed
 for k in 257:n;expected[k]=expected[k-55]+expected[k-119]+expected[k-179]+expected[k-256];end
 x=CUDA.zeros(UInt32,n);y=CUDA.zeros(UInt32,768);streams=(CuStream(),CuStream());workspaces=nothing
 function reset!()
  fill!(x,0);copyto!(view(x,1:256),seed);fill!(y,0);CUDA.@allowscalar y[513]=1;CUDA.synchronize()
 end
 reset!();pipeline!(x,y,s,r,streams,workspaces)
 for trial in 1:5
  reset!();start=time_ns();pipeline!(x,y,s,r,streams,workspaces);seconds=(time_ns()-start)*1e-9
  result=Array(x);correct=result==expected;error=correct ? 0 : 1
  artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",n,",\"max_abs_error\":",error,",\"timing_region\":\"four_stage_parallel_pipeline_with_stream_joins\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"uint32\",\"count\":",n,",\"byte_order\":\"little\"}}")
  correct||return 1
 end
 return 0
end
exit(main(ARGS))
