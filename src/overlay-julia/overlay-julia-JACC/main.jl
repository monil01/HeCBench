using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function overlay_kernel(ix,iy,input,output,W,H,x0,y0,bw,bh,color)
 overlay_pixel(input,output,W,H,x0,y0,bw,bh,color,Int32(ix-1),Int32(iy-1))
end
function factory(input,output,boxes,color)
 spec=JACC.launch_spec(;threads=(8,8),blocks=(1,1),shmem_size=0,sync=false)
 ()->begin
  for box in boxes
   extent=(cld(Int(box.width),8)*8,cld(Int(box.height),8)*8)
   # LaunchSpec caches blocks; original rectangle grids vary on every call.
   spec.blocks=(extent[1]÷8,extent[2]÷8)
   JACC.parallel_for(spec,extent,overlay_kernel,input,output,Int32(640),Int32(480),box.left,box.top,box.width,box.height,color)
  end
 end
end
main()=overlay_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
