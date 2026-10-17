using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function overlay_kernel(input,output,W,H,x0,y0,bw,bh,color)
 ix,iy=@index(Global,NTuple)
 overlay_pixel(input,output,W,H,x0,y0,bw,bh,color,Int32(ix-1),Int32(iy-1))
end
function factory(input,output,boxes,color)
 kernel=overlay_kernel(CUDABackend(),(8,8))
 ()->begin
  for box in boxes
   kernel(input,output,Int32(640),Int32(480),box.left,box.top,box.width,box.height,color;ndrange=(cld(Int(box.width),8)*8,cld(Int(box.height),8)*8))
  end
 end
end
main()=overlay_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
