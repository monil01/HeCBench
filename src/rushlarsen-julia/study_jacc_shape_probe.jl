using CUDA
import JACC
JACC.@init_backend
function fillindex!(i,a)
 a[i]=Int32(i)
end
spec=JACC.launch_spec(threads=256,shmem_size=0,sync=false)
for n in (33,257)
 a=JACC.zeros(Int32,n);fill!(a,Int32(-99))
 JACC.parallel_for(spec,n,fillindex!,a);CUDA.synchronize()
 h=Array(a)
 println("SHAPE_PROBE n=",n," blocks=",spec.blocks," correct=",count(i->h[i]==i,1:n)," tail=",h[end])
end
