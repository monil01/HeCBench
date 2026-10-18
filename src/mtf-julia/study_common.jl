using CUDA
include("study_fixtures.jl")
function independent(word)
 last=fill(-1,256);ranks=UInt8[]
 for(i,w)in enumerate(word)
  j=last[Int(w)+1];rank=j<0 ? Int(w)+count(a->last[a+1]>=0,Int(w)+1:255) : count(>(j),last);push!(ranks,UInt8(rank));last[Int(w)+1]=i
 end
 order=sort(collect(0:255);by=a->(last[a+1]<0,last[a+1]<0 ? a : -last[a+1]));return ranks,UInt8.(order)
end
function decoder(ranks)
 a=collect(UInt8(0):UInt8(255));out=UInt8[]
 for r in ranks;j=Int(r)+1;w=a[j];push!(out,w);deleteat!(a,j);pushfirst!(a,w);end
 return out
end
function study(allocate,search)
 cases=[];prefix=UInt32[]
 for(id,word)in enumerate(FIXTURES)
  n=length(word);hraw=fill(UInt8(0xa5),320);oraw=fill(UInt8(0xa5),n+64);wraw=fill(UInt8(0xa5),n+64);copyto!(wraw,33,word,1,n);d_raw=allocate(UInt8,320);fill!(d_raw,UInt8(0xa5));d=view(d_raw,33:288);r_raw=allocate(Int32,65);fill!(r_raw,Int32(0x5a5a5a5a));rank=view(r_raw,33:33);reference=independent(word);append!(prefix,UInt32[id-1,n,1]);append!(prefix,UInt32.(wraw[33:n+32]));push!(cases,(n,hraw,oraw,wraw,d_raw,d,r_raw,rank,reference))
 end
 function run(c)
  n,hraw,oraw,wraw,d_raw,d,r_raw,rank,ref=c;CUDA.device_synchronize(blocking=true);start=time_ns()
  for j in 0:255;hraw[j+33]=UInt8(j);end
  for i in 1:n
   copyto!(d,1,hraw,33,256);search(d,wraw[i+32],rank);j=Int(Array(rank)[1]);0<=j<256||error("AlphabetFind");oraw[i+32]=UInt8(j)
   if j>0;copyto!(hraw,34,d,1,j);hraw[33]=wraw[i+32];end
  end
  CUDA.device_synchronize(blocking=true);t=(time_ns()-start)*1e-9;encoded=oraw[33:n+32];alpha=hraw[33:288];ok=encoded==ref[1]&&alpha==ref[2]&&decoder(encoded)==wraw[33:n+32]
  guards=true
  for raw in(hraw,oraw,wraw);guards&=all(==(UInt8(0xa5)),raw[1:32])&&all(==(UInt8(0xa5)),raw[end-31:end]);end
  guards&=all(==(UInt8(0xa5)),Array(view(d_raw,1:32)))&&all(==(UInt8(0xa5)),Array(view(d_raw,289:320)));guards&=all(==(Int32(0x5a5a5a5a)),Array(view(r_raw,1:32)))&&all(==(Int32(0x5a5a5a5a)),Array(view(r_raw,34:65)))
  return vcat(UInt32.(encoded),UInt32.(alpha)),t,ok&&guards,guards
 end
 for c in cases;run(c);end
 for trial in 1:5
  out=copy(prefix);time=0.;ok=true;ts=Float64[]
  for(id,c)in enumerate(cases);v,t,pass,guards=run(c);append!(out,v);time+=t;ok&=pass;push!(ts,t);println("STUDY_GUARDS trial=",trial," case=",id-1," exact=",guards);end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"mtf_$(trial).bin");open(io->write(io,out),file,"w");println("STUDY_COMPONENT_SECONDS ",join(ts,","));println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",time,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum9_preallocated_corrected_online_hostdriven_mtf\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",file,"\",\"dtype\":\"uint32\",\"count\":",length(out),",\"byte_order\":\"little\"}}");ok||error("FullOnlineMTF")
 end
end
