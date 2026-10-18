@inline function fsm_random!(rng,id)
 @inbounds v=(UInt32(26757677)*rng[id]+UInt32(1))&UInt32(0x7fffffff);rng[id]=v
 return v
end
