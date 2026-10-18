using CUDA
include("../study_device.jl")
include("../study_common.jl")
@inline model_atomic!(a,i,v)=CUDA.atomic_add!(pointer(a,i),v)
function match!(a,b,c);g=(blockIdx().x-1)*blockDim().x+threadIdx().x;if g<=768;match_lane!(g,a,b,c);end;return;end
function gather!(a,b,c,d);t=(blockIdx().x-1)*blockDim().x+threadIdx().x;if t<=49152;gather_lane!(t,a,b,c,d);end;return;end
function transform!(a,inverse);t=(blockIdx().x-1)*blockDim().x+threadIdx().x;if t<=12288;transform_lane!(t,a,inverse);end;return;end
function hard!(a,b,c);g=(blockIdx().x-1)*blockDim().x+threadIdx().x;if g<=768;hard_lane!(g,a,b,c);end;return;end
function aggregate!(a,b,c,d,e,f,g);t=(blockIdx().x-1)*blockDim().x+threadIdx().x;if t<=49152;aggregate_lane!(t,a,b,c,d,e,f,g);end;return;end
function final!(a,b,c);t=(blockIdx().x-1)*blockDim().x+threadIdx().x;if t<=4480;final_lane!(t,a,b,c);end;return;end
function factory(a)
 ()->begin
  @cuda threads=64 blocks=12 match!(a.img,a.stacks,a.counts)
  @cuda threads=128 blocks=384 gather!(a.img,a.stacks,a.counts,a.group)
  @cuda threads=128 blocks=96 transform!(a.group,false)
  @cuda threads=64 blocks=12 hard!(a.group,a.counts,a.weights)
  @cuda threads=128 blocks=96 transform!(a.group,true)
  @cuda threads=128 blocks=384 aggregate!(a.group,a.stacks,a.counts,a.weights,a.win,a.num,a.den)
  @cuda threads=128 blocks=35 final!(a.num,a.den,a.out)
 end
end
bm_study(factory)
