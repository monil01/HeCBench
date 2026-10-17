using CUDA,Printf
function mc_read(name,T)
 data=read(joinpath(@__DIR__,name));reinterpret(T,data)|>collect
end
function marching_study(factory)
 expected=mc_read("study_expected_f64.bin",Float64);prefix=mc_read("study_input_prefix_f64.bin",Float64);mm2cpu=reshape(mc_read("study_minmax2_f64.bin",Float64),2,340,144)
 d=CuArray(mc_read("study_tables_u16.bin",UInt16));t=CuArray(mc_read("study_tables_i32.bin",Int32));e=CuArray(mc_read("study_edges_u8.bin",UInt8));iso=Float32(-.9)
 mm1=CUDA.zeros(Float32,147968);ids1=CUDA.zeros(UInt32,73984);mm2=CUDA.zeros(Float32,8296*225*2);ids2=CUDA.zeros(UInt32,8296*225);n1d=CUDA.zeros(UInt32,1);n2d=similar(n1d);counts=CUDA.zeros(UInt32,2);checksum=CUDA.zeros(UInt64,1);coords=CUDA.zeros(Float32,4)
 ops=factory(mm1,ids1,mm2,ids2,n1d,n2d,counts,checksum,coords,d,t,e,iso)
 function preprocess!()
  fill!(n1d,0);fill!(n2d,0);fill!(counts,0);fill!(checksum,0);fill!(coords,0);ops[1]();ops[2]();n1=Int(only(Array(n1d)));n1==8296||error("Lv1count $(n1)");ops[3](n1);ops[4](n1);n2=Int(only(Array(n2d)));n2==240380||error("Lv2count $(n2)");n1,n2
 end
 n1,n2=preprocess!();launch=ops[5](n2);launch();CUDA.synchronize() # full-pipeline/JIT warmup, not measured
 allok=true
 for trial in 1:5
  prepstart=time_ns();n1,n2=preprocess!();CUDA.synchronize();prepseconds=(time_ns()-prepstart)*1e-9
  start=time_ns();launch();CUDA.synchronize();seconds=(time_ns()-start)*1e-9
  h1=Array(ids1)[1:n1];h2=Array(ids2)[1:n2];hmm=Array(mm2);maxmm=0.
  for parent in 0:n1-1
   id=Int(h1[parent+1]);bx=id%68;by=(id÷68)%68;bz=id÷4624
   for sub in 0:224,k in 1:2
    x=bx*5+sub%5;z=bz*9+sub÷25;index=2*(225parent+sub)+k;err=abs(Float64(hmm[index])-mm2cpu[k,x+1,z+1]);maxmm=max(maxmm,err)
   end
  end
  isfinite(maxmm)&&maxmm<1e-3||error("full Lv2 minmax CPU mismatch $(maxmm)");hc=Array(counts);got=vcat(prefix,Float64.(Array(mm1)),Float64.(sort(h1)),Float64.(sort(h2)),[n1,n2,hc[1],hc[2]],Float64.(Array(coords))./Float64(hc[1]));length(got)==length(expected)||error("shape")
  errorabs=maximum(abs.(got.-expected));ok=all(isfinite,got)&&errorabs<=1e-3;allok&=ok
  println("STUDY_PIPELINE_PROPERTIES untimed_preprocessing_seconds=",prepseconds," full_lv2_minmax_checked=",length(hmm)," max_abs_error=",maxmm," scheduling_dependent_triangle_index_checksum=",only(Array(checksum))," normalized_partial_coordinate_sums=",got[end-3:end]," mesh_arrays_not_produced=true")
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,got)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(got),errorabs,errorabs/1e-3,path,length(got))
 end
 allok||error("CPU output mismatch");println("PASS")
end
