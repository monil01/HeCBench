const SHMEM_VEC4=NTuple{4,VecElement{Float32}}
@inline function shmem_init(i::Int32)
 (VecElement(Float32(i)),VecElement(Float32(i+Int32(11))),VecElement(Float32(i+Int32(19))),VecElement(Float32(i+Int32(23))))
end
@inline function shmem_swap!(shared,a,b)
 @inbounds temp=shared[b];shared[b]=shared[a];shared[a]=temp
 return
end
@inline function shmem_reduce(shared,tid)
 @inbounds a=shared[tid+1];b=shared[tid+257];c=shared[tid+513];d=shared[tid+769];e=shared[tid+1025];f=shared[tid+1281]
 ntuple(Val(4)) do k
  VecElement(a[k].value+b[k].value+c[k].value+d[k].value+e[k].value+f[k].value)
 end
end
