using CUDA
include("../study_common.jl")
function overlay_kernel(input,output,W,H,x0,y0,bw,bh,color)
 bx=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 by=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y-Int32(1)
 overlay_pixel(input,output,W,H,x0,y0,bw,bh,color,bx,by)
 return
end
function factory(input,output,boxes,color)
 ()->begin
  for box in boxes
   @cuda threads=(8,8) blocks=(cld(Int(box.width),8),cld(Int(box.height),8)) overlay_kernel(input,output,Int32(640),Int32(480),box.left,box.top,box.width,box.height,color)
  end
 end
end
main()=overlay_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
