using CUDA,JACC
const P1=55;const P2=119;const P3=179;const P4=256
function first_column!(index,x,s,workspace)
 t=mod(index-1,256);cx=JACC.shared(workspace)
 cx[t+1]=x[t+1]
 JACC.sync_workgroup()
 for k in 1:div(s,256)-1
  for i in 0:32:255
   if t<32
    j=256+i+t;cx[j+1]=cx[j-55+1]+cx[j-119+1]+cx[j-179+1]+cx[j-256+1]
   end
   JACC.sync_workgroup()
  end
  cx[t+1]=cx[256+t+1];x[k*256+t+1]=cx[t+1]
  JACC.sync_workgroup()
 end
 return nothing
end
function column_y!(index,y,s,workspace)
 t=mod(index-1,256);cy=JACC.shared(workspace)
 # Native initially reads undefined lower half, dead after >=2rounds; initialize for definedportableexecution.
 cy[t+1]=0;cy[256+t+1]=0;cy[512+t+1]=y[512+t+1]
 JACC.sync_workgroup()
 for k in 0:div(s,256)-1
  cy[t+1]=cy[256+t+1];cy[256+t+1]=cy[512+t+1]
  JACC.sync_workgroup()
  for i in 0:32:255
   if t<32
    j=512+i+t;cy[j+1]=cy[j-55+1]+cy[j-119+1]+cy[j-179+1]+cy[j-256+1]
   end
   JACC.sync_workgroup()
  end
 end
 y[t+1]=cy[t+1];y[256+t+1]=cy[256+t+1];y[512+t+1]=cy[512+t+1]
 return nothing
end
function last_entries!(index,x,y,s,r,workspace)
 t=index-1;mem=JACC.shared(workspace)
 # a0[0:767], b0[768:1279], c0[1280:1791], d0[1792:2303]
 mem[t+1]=y[t+1]
 JACC.sync_workgroup()
 if t<256;mem[t+512+1]=y[t+512+1];end
 JACC.sync_workgroup()
 mem[1792+t+1]=mem[256+t+1];mem[1280+t+1]=mem[256+t+1];mem[768+t+1]=mem[256+t+1]
 JACC.sync_workgroup()
 mem[768+t+1]+=mem[256-77+t+1]
 JACC.sync_workgroup()
 mem[1280+t+1]+=mem[256-60+t+1]+mem[256-137+t+1]
 JACC.sync_workgroup()
 mem[1792+t+1]+=mem[256-64+t+1]+mem[256-124+t+1]+mem[256-201+t+1]
 JACC.sync_workgroup()
 for column in 1:r-1
  tmp=UInt32(0)
  if t<256
   for k in 0:76;tmp+=x[column*s-256+k+1]*mem[512+t-k+1];end
   for k in 0:59;tmp+=x[column*s-179+k+1]*mem[1024+t-k+1];end
   for k in 0:63;tmp+=x[column*s-119+k+1]*mem[1536+t-k+1];end
   for k in 0:54;tmp+=x[column*s-55+k+1]*mem[2048+t-k+1];end
   x[column*s+s-256+t+1]=tmp
  end
  JACC.sync_workgroup()
 end
 return nothing
end
function columns!(index,x,s,r,workspace)
 t=mod(index-1,256);block=div(index-1,256);warp=div(t,32);lane=t%32;cx=JACC.shared(workspace)
 # Initialize dead first-column workspace; native leaves it undefined but never exports that row.
 for i in 0:7;cx[i*512+t+1]=0;cx[i*512+256+t+1]=0;end
 first=block==0 ? 1 : 0;last=(block==cld(r,8)-1&&r%8!=0) ? r%8 : 8
 for i in first:last-1;cx[i*512+t+1]=x[(block*8+i)*s-256+t+1];end
 JACC.sync_workgroup()
 for k in 0:div(s,256)-2
  for i in 0:32:255
   if !(block==0&&t==0)&&!(block==cld(r,8)-1&&warp>=last)
    j=warp*512+256+i+lane;cx[j+1]=cx[j-55+1]+cx[j-119+1]+cx[j-179+1]+cx[j-256+1]
   end
   JACC.sync_workgroup()
  end
  for i in first:last-1
   cx[i*512+t+1]=cx[i*512+256+t+1];x[(block*8+i)*s+k*256+t+1]=cx[i*512+t+1]
  end
  JACC.sync_workgroup()
 end
 return nothing
end
function pipeline!(x,y,s,r,streams,workspaces)
 dx=CUDA.cudaconvert(x);dy=CUDA.cudaconvert(y)
 GC.@preserve x y workspaces begin
  CUDA.stream!(streams[1]) do
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=1,shmem_size=2048,sync=false,stream=streams[1]),256,first_column!,dx,s,workspaces[1])
  end
  CUDA.stream!(streams[2]) do
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=1,shmem_size=3072,sync=false,stream=streams[2]),256,column_y!,dy,s,workspaces[2])
  end
  CUDA.synchronize(streams[1]);CUDA.synchronize(streams[2])
  JACC.parallel_for(JACC.launch_spec(threads=512,blocks=1,shmem_size=9216,sync=false),512,last_entries!,dx,dy,s,r,workspaces[3])
  JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(r,8),shmem_size=16384,sync=false),cld(r,8)*256,columns!,dx,s,r,workspaces[4])
  CUDA.synchronize()
 end
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 16384;r=16;s=div(n,r);s%256==0&&s>=512||error("fixture shape")
 seed=UInt32[UInt32(i*1664525+1013904223) for i in 0:255];expected=zeros(UInt32,n);expected[1:256]=seed
 for k in 257:n;expected[k]=expected[k-55]+expected[k-119]+expected[k-179]+expected[k-256];end
 x=CUDA.zeros(UInt32,n);y=CUDA.zeros(UInt32,768);streams=(CuStream(),CuStream());workspaces=Tuple(CUDA.zeros(UInt32,n) for n in (512,768,2304,4096))
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
