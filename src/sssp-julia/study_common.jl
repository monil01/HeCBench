const G=Int32(-19371337);const NG=17;const INF=Int32(-2147483647);const WHITE=Int32(16677217);const GRAY0=Int32(16677219);const GRAY1=Int32(16677220)
mutable struct Buffer
 h::Vector{Int32};d::CuArray{Int32,1};n::Int
end
function Buffer(n,value)
 h=vcat(fill(Int32(value),n),fill(G,NG));Buffer(h,CuArray(h),n)
end
up!(a)=(copyto!(a.d,a.h);nothing)
down!(a)=(copyto!(a.h,a.d);nothing)
getdev(a)=Array(a.d)
guarded(a)=all(==(G),@view getdev(a)[a.n+1:length(a.h)])
readints(file)=collect(reinterpret(Int32,read(file)))
writeints(path,x)=open(io->write(io,x),path,"w")
function study_main()
 @assert length(ARGS)==5;name=ARGS[1];limit=parse(Int,ARGS[2]);blocks=parse(Int,ARGS[3]);expect_gpu=ARGS[4]=="1";auxbase=ARGS[5];raw=readints(name*"_graph_i32.bin");n,e,source=Int.(raw[1:3]);@assert length(raw)==3+2n+2e;oracle=readints(name*"_oracle_i32.bin");@assert length(oracle)==n
 CUDA.stream!(CUDA.legacy_stream());nodes=Buffer(2n,0);edges=Buffer(2e,0);cost=Buffer(n,INF);color=Buffer(n,WHITE);q1=Buffer(n,0);q2=Buffer(n,0);sc=Buffer(8,0);copyto!(nodes.h,1,raw,4,2n);copyto!(edges.h,1,raw,4+2n,2e);up!(nodes);up!(edges);nodes0=copy(nodes.h);edges0=copy(edges.h);folder=joinpath(auxbase,"pid$(getpid())");mkpath(folder)
 lib=Libdl.dlopen(abspath("libsssp_host.so"));firstfp=Libdl.dlsym(lib,:sssp_first);hostfp=Libdl.dlsym(lib,:sssp_host_level)
 function first!()
  ccall(firstfp,Cvoid,(Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32}),nodes.h,edges.h,cost.h,color.h,q1.h,q2.h,sc.h)
 end
 function hostlevel!()
  ccall(hostfp,Cvoid,(Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Int32),nodes.h,edges.h,cost.h,color.h,q1.h,q2.h,sc.h,Int32(limit))
 end
 state()=vcat(getdev(cost),getdev(color),getdev(q1),getdev(q2),getdev(sc))
 function validfrontier()
  @assert 0<=sc.h[1]<=n && sc.h[4]==0
  q=iseven(sc.h[6]) ? q1 : q2
  @assert all(x->0<=x<n,@view q.h[1:Int(sc.h[1])])
 end
 for trial=0:5
  fill!(@view(cost.h[1:n]),INF);cost.h[source+1]=0;fill!(@view(color.h[1:n]),WHITE);fill!(@view(q1.h[1:n]),0);q1.h[1]=source;fill!(@view(q2.h[1:n]),0);fill!(@view(sc.h[1:8]),0);sc.h[1]=1;sc.h[5]=GRAY0
  for x in (cost,color,q1,q2,sc);up!(x);end
  CUDA.device_synchronize(blocking=true);prefix=vcat(getdev(nodes),getdev(edges));@assert state()==vcat(cost.h,color.h,q1.h,q2.h,sc.h)
  firstgpu=Int32[];seconds=0.0;cpusec=0.0;gpusec=0.0;gpucalls=0;cpucalls=0;maxfrontier=0;lastgpu=false;t=time_ns();first!();z=(time_ns()-t)*1e-9;seconds+=z;cpusec+=z
  while sc.h[1]!=0
   validfrontier();maxfrontier=max(maxfrontier,Int(sc.h[1]))
   if sc.h[1]<limit
    t=time_ns()
    while sc.h[1]!=0 && sc.h[1]<limit
     maxfrontier=max(maxfrontier,Int(sc.h[1]));hostlevel!();cpucalls+=1;@assert 0<=sc.h[1]<=n
    end
    z=(time_ns()-t)*1e-9;seconds+=z;cpusec+=z;lastgpu=false
   else
    for x in (cost,color,q1,q2,sc);up!(x);end
    CUDA.device_synchronize(blocking=true)
    while sc.h[1]!=0 && sc.h[1]>=limit
     maxfrontier=max(maxfrontier,Int(sc.h[1]));validfrontier();up!(sc);CUDA.device_synchronize(blocking=true)
     isempty(firstgpu)&&(firstgpu=state());qin,qout=iseven(sc.h[6]) ? (q1,q2) : (q2,q1);active=Array(view(qin.d,1:Int(sc.h[1])));@assert all(x->0<=x<n,active)
     t=time_ns();launch!(nodes.d,edges.d,cost.d,color.d,qin.d,qout.d,sc.d,blocks);CUDA.device_synchronize(blocking=true);z=(time_ns()-t)*1e-9;seconds+=z;gpusec+=z;gpucalls+=1;down!(sc);@assert sc.h[4]==0;sc.h[1]=sc.h[3];sc.h[3]=0;sc.h[2]=0;sc.h[5]=iseven(sc.h[6]) ? GRAY0 : GRAY1;lastgpu=true
    end
    for x in (cost,color,q1,q2);down!(x);end
   end
   @assert sc.h[6]<1000000
  end
  @assert !isempty(firstgpu);append!(prefix,firstgpu);ok=(@view(cost.h[1:n]))==oracle && sc.h[4]==0 && (!expect_gpu||lastgpu);gpu_final=state();ok &= getdev(nodes)==nodes0 && getdev(edges)==edges0 && all(guarded,(cost,color,q1,q2,sc));expect_gpu&&(ok &= (@view(gpu_final[1:n]))==oracle)
  out=copy(cost.h);aux=vcat(gpu_final,color.h,q1.h,q2.h,sc.h);auxpath=joinpath(folder,"trial$(trial)_private.bin");path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin")
  if trial>0 || !ok
   writeints(auxpath,aux);writeints(path,vcat(prefix,out));ae=ok ? "0" : "null";tv=trial>0 ? "true" : "false";gpuclaim=expect_gpu&&ok ? "true" : "null"
   println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$seconds,\"correct\":$ok,\"checked_values\":$(length(out)),\"max_abs_error\":$ae,\"max_rel_error\":null,\"timing_region\":\"sum_source_CPU_first_CPU_worker_and_completed_GPU_kernel_segments_full_convergence_excluding_transfers_control_validation\",\"timing_valid\":$tv,\"jit_excluded\":$tv,\"input_prefix_count\":$(length(prefix)),\"CPU_seconds\":$cpusec,\"GPU_seconds\":$gpusec,\"CPU_worker_frontiers\":$cpucalls,\"GPU_kernel_launches\":$gpucalls,\"total_frontier_levels\":$(sc.h[6]),\"max_frontier\":$maxfrontier,\"last_stage_GPU\":$lastgpu,\"final_GPU_cost_exact_independent_Dijkstra\":$gpuclaim,\"raw_private_artifact\":{\"path\":\"$auxpath\",\"dtype\":\"int32\",\"count\":$(length(aux)),\"scope\":\"actual_last_GPU_all_mutable_buffers_then_hostfinal_color_q1_q2_scalars;queueorder_not_canonicalized_not_raw_identity_claim\"},\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"int32\",\"count\":$(length(prefix)+length(out))}}")
   flush(stdout)
  end
  ok||exit(1)
 end
 for x in (nodes,edges,cost,color,q1,q2,sc);CUDA.unsafe_free!(x.d);end
 Libdl.dlclose(lib)
end
