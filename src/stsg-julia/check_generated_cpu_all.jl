include("filter_device.jl")
fixture=reinterpret(Float64,read(joinpath(@__DIR__,"study_fixture.bin")))
gold=reinterpret(Float64,read(joinpath(@__DIR__,"study_oracle.bin")))
function check_all(fixture,gold)
 off=2;got=Float64[]
 for ci in 1:Int(fixture[1])
  X,Y,B,Z,w=Int32.(fixture[off:off+4]);cosy=Float32(fixture[off+5]);samp=Float32(fixture[off+6]);snow=Int32(fixture[off+7]);off+=8;N=X*Y;W=(2w+1)^2;Q=N*B*Z
  raw=Int16.(fixture[off:off+Q-1]);off+=Q;rawqa=UInt8.(fixture[off:off+Q-1]);off+=Q
  nd=zeros(Float32,Q);qa=zeros(Float32,Q);ref=zeros(Float32,N*B);res=zeros(Float32,N*W*4);r3=zeros(Float32,N*W*3);out=zeros(Float32,Q);vin=zeros(Float32,N*B);res1=zeros(Int32,N*B);index=zeros(Int32,N*W*3)
  eachcell(call)=foreach(t->call(UInt32(t[1]),UInt32(t[2]),UInt32(1),UInt32(1),UInt32(0),UInt32(0)),((i,j) for j in 0:Y-1 for i in 0:X-1))
  eachcell((ids...)->Short_to_Float_body!(raw,rawqa,X,Y,B,Z,nd,qa,ids...));append!(got,nd);append!(got,qa)
  eachcell((ids...)->Generate_NDVI_reference_body!(cosy,Int32(2),nd,qa,X,Y,B,Z,ref,r3,res1,ids...));for v in (nd,qa,ref,r3,res1);append!(got,v);end
  fill!(r3,0f0)
  eachcell((ids...)->Compute_d_res_body!(nd,qa,ref,Int32(0),Y,Int32(0),Int32(0),X,Y,B,Z,w,res,ids...));append!(got,res)
  eachcell((ids...)->STSG_filter_body!(nd,qa,ref,Int32(0),Y,Int32(0),Int32(0),X,Y,B,Z,w,samp,snow,out,vin,res,r3,index,ids...))
  for v in (nd,qa,ref,res,r3,out,vin,res1,index);append!(got,v);end
 end
 @assert off==length(fixture)+1 && length(got)==length(gold)
 err=maximum(abs.(got.-gold));bad=count(i->!isfinite(got[i])||abs(got[i]-gold[i])>2e-5+2e-5*abs(gold[i]),eachindex(gold))
 println("STSG_GENERATED_CPU_CHECK values=",length(got)," max_abs_error=",err," violations=",bad);@assert bad==0
end
check_all(fixture,gold)
