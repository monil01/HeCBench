using CUDA,Printf
const H_FIELDS=(:slack,:min_in_rows,:min_in_cols,:zeros,:zeros_size_b,:row_of_star_at_column,:column_of_star_at_row,:cover_row,:cover_column,:column_of_prime_at_row,:row_of_green_at_column,:d_min_in_mat_vect,:d_min_in_mat)
const H_SIZES=(65536,256,256,65536,1,256,256,256,256,256,256,256,1)
function hsolve(launch!,S,hostflags)
 counts=zeros(Int,16);launches=0
 function run(k,nb,nt,shared=0)
  launch!(Val(k),S,nb,nt,shared);CUDA.device_synchronize(blocking=true);counts[k+1]+=1;launches+=1;launches<=10000||error("iterationcap")
 end
 run(0,4,64);run(1,256,256,1024);run(2,256,256);run(3,256,256,1024);run(4,256,256);run(5,256,256)
 while true
  hostflags[4]=0;run(6,1,min(256,Int(hostflags[1])),12);hostflags[4]==0&&break
 end
 while true
  run(7,4,64);run(8,4,64);hostflags[2]>=256&&break;run(9,4,64)
  while true
   while true
    hostflags[3]=0;hostflags[4]=0;run(10,1,min(256,Int(hostflags[1])),12)
    !(hostflags[4]!=0&&hostflags[3]==0)&&break
   end
   hostflags[3]!=0&&break
   run(13,256,256,1024);run(14,1,128,512);run(15,256,256);run(5,256,256)
  end
  run(11,4,64);run(12,4,64)
 end
 counts
end
function hungarian_study(launch!)
 prefix=collect(reinterpret(Int32,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Int32,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(prefix)==196610&&length(gold)==198147
 owners=[CUDA.fill(Int32(1515870810),n+16) for n in H_SIZES];views=[view(d,9:length(d)-8) for d in owners];flags=CuArray{Int32,1,CUDA.UnifiedMemory}(undef,4);hostflags=unsafe_wrap(Array,convert(Ptr{Int32},flags.data[].mem),(4,);own=false);fill!(hostflags,0)
 # Raw device view prevents automatic managed-memory prefetch at every kernel launch.
 dflags=Base.unsafe_convert(CUDA.CuDeviceArray{Int32,1,CUDA.AS.Global},flags);S=merge(NamedTuple{H_FIELDS}(Tuple(views)),(flags=dflags,))
 costs=[prefix[3+ci*65536:2+(ci+1)*65536] for ci in 0:2]
 for c in costs;copyto!(S.slack,c);hsolve(launch!,S,hostflags);end
 for trial in 1:5
  art=copy(prefix);raw=Int32[];times=Float64[];counts=Int[];ok=true
  for c in costs
   copyto!(S.slack,c);CUDA.device_synchronize(blocking=true);t=time_ns();co=hsolve(launch!,S,hostflags);CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-t)*1e-9);append!(counts,co)
   cs=Array(S.column_of_star_at_row);rs=Array(S.row_of_star_at_column);sl=Array(S.slack);append!(raw,sl);append!(raw,cs);append!(raw,rs);ok&=sort(cs)==Int32.(0:255)&&sort(rs)==Int32.(0:255)&&all(>=(0),sl);total=Int64(0)
   for r in 0:255
    col=cs[r+1]
    if 0<=col<256;total+=c[col*256+r+1];ok&=rs[col+1]==r&&sl[col*256+r+1]==0;else;ok=false;end
   end
   append!(art,cs);append!(art,rs);push!(art,Int32(total));append!(art,c)
   for d in owners
    h=Array(d);ok&=all(==(Int32(1515870810)),h[1:8])&&all(==(Int32(1515870810)),h[end-7:end])
   end
  end
  ok&=art[196610+1:end]==gold;@assert length(art)==394757;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art);aux=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"raw_slack_assignment_trial$(trial).bin");write(aux,raw)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":198147,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum3_full_Hungarian_host_controlled_GPU_solvers\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":394757,\"byte_order\":\"little\"},\"auxiliary_artifacts\":[{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":198144}],\"path_times_seconds\":%s,\"stage_launch_counts\":%s}\n",trial,sum(times),ok ? "true" : "false",file,aux,string(times),string(counts))
 end
end
