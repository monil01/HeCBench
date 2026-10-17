using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
@inline histogram_bin(x::UInt8)=Int32(x)
@inline histogram_bin(x::Float32)=trunc(Int32,x*256.f0)
function histogram_study(factory)
 method=isempty(ARGS) ? "gmem" : ARGS[1];kind=length(ARGS)<2 ? "u4" : ARGS[2];channels=kind=="u1" ? 1 : 3;stride=kind=="u1" ? 1 : 4;bins=channels*256
 width=1080;height=1920;n=width*height;rgba=Vector{UInt8}(undef,4n);state=UInt64(0xabcdef)
 for i in eachindex(rgba)
  state=state*UInt64(6364136223846793005)+UInt64(1442695040888963407);rgba[i]=UInt8((state>>32)&UInt64(255))
 end
 host=kind=="u1" ? UInt8[div(Int(rgba[4i+1])+Int(rgba[4i+2])+Int(rgba[4i+3]),3) for i in 0:n-1] : kind=="f4" ? Float32.(rgba)./256.f0 : rgba
 final=zeros(Int32,bins);partial=zeros(Int32,bins*256)
 for row in 0:height-1,col in 0:width-1
  group=div(mod(col,512),32)+16*div(mod(row,64),4);pixel=(row*width+col)*stride
  for channel in 0:channels-1
   bin=channel*256+Int(histogram_bin(host[pixel+channel+1]))+1;final[bin]+=1;partial[group*bins+bin]+=1
  end
 end
 for ch in 0:channels-1;sum(final[ch*256+1:(ch+1)*256])==n||error("histogram conservation");end
 expected=vcat(final,partial);input=CuArray(host);out=CUDA.zeros(UInt32,bins);parts=CUDA.zeros(UInt32,256*1024);operation! =factory(input,parts,out,width,height,channels,stride,method)
 reset! =()->begin;fill!(parts,UInt32(0));fill!(out,UInt32(0));end
 readback=()->begin
  po=Array(parts);active=Int32[po[g*1024+i] for g in 0:255 for i in 1:bins];vcat(Int32.(Array(out)),active)
 end
 println("STUDY_CONTRACT method=",method," kind=",kind," kernel_view1080x1920 source_driver_default_shape_swapped_into_TestMethods bins=",bins," active_partial_counts=",256*bins," final_counts=",bins," all3color_channels_conserved entropy_presets_and_fileparser_unmeasured")
 OwnedStudy.run_trials!(operation!,reset!,readback,expected,10;atol=0.,rtol=0.)
end
