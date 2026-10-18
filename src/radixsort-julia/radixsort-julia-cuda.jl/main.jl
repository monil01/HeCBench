include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function run(keys,tmp,counts,blocks,scan,buffer,n)
 nb=n÷Int32(256);cn=n÷Int32(16);bn=cn÷Int32(1024)
 for startbit in Int32(0):Int32(4):Int32(28)
  @cuda threads=128 blocks=n÷Int32(512) local_sort_cuda!(keys,tmp,startbit)
  @cuda threads=128 blocks=nb offsets_cuda!(tmp,counts,blocks,startbit,nb)
  @cuda threads=256 blocks=bn scan1_cuda!(scan,counts)
  @cuda threads=256 blocks=1 scan2_cuda!(buffer,scan,counts,bn)
  @cuda threads=256 blocks=bn uniform_cuda!(scan,buffer)
  @cuda threads=128 blocks=nb reorder_cuda!(keys,tmp,blocks,scan,startbit,nb,n)
 end
 return
end
study(run,(T,n)->CuArray{T}(undef,n))
