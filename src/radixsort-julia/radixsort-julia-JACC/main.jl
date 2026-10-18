include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const TEMPLATE512=JACC.zeros(UInt32,512)
const TEMPLATE272=JACC.zeros(UInt32,272)
const TEMPLATE288=JACC.zeros(UInt32,288)
const TEMPLATE1=JACC.zeros(UInt32,1)
function launch(f,N,threads,template,args...)
 JACC.parallel_for(JACC.launch_spec(threads=threads,blocks=N÷threads,shmem_size=4length(template),sync=false),N,f,args...,template);return
end
function run(keys,tmp,counts,blocks,scan,buffer,n)
 nb=n÷Int32(256);bn=n÷Int32(16384)
 for startbit in Int32(0):Int32(4):Int32(28)
  launch(local_sort_jacc!,Int(n)÷4,128,TEMPLATE512,keys,tmp,startbit)
  launch(offsets_jacc!,Int(n)÷2,128,TEMPLATE272,tmp,counts,blocks,startbit,nb)
  launch(scan1_jacc!,Int(bn)*256,256,TEMPLATE512,scan,counts)
  launch(scan2_jacc!,256,256,TEMPLATE512,buffer,scan,counts,bn)
  launch(uniform_jacc!,Int(bn)*256,256,TEMPLATE1,scan,buffer)
  launch(reorder_jacc!,Int(n)÷2,128,TEMPLATE288,keys,tmp,blocks,scan,startbit,nb,n)
 end
 return
end
study(run,(T,n)->JACC.zeros(T,n))
