struct StudyFour
 v::Float32
 x::Float32
 y::Float32
 z::Float32
end
@inline box_offset(meta,bx)=reinterpret(Int64,meta[Int64(bx)*82+3])
@inline box_nn(meta,bx)=reinterpret(Int32,UInt32(meta[Int64(bx)*82+4]&UInt64(0xffffffff)))
@inline box_neighbor(meta,bx,k)=reinterpret(Int32,UInt32(meta[Int64(bx)*82+6+Int64(k)*3]>>32))
