# New reference-phase translation of original reference.h GPU formulas.
@inline function sigmoid_term(l::Float32,t::Float32,mode::Int32)
 pos=Float32(l>=0f0)
 logterm=log(1f0+exp(l-2f0*l*pos))
 if mode==2
  return l*t+(t-1f0)*l*pos-(1f0-t)*logterm
 elseif mode==1
  partition=l*pos+logterm
  return (2f0*t-1f0)*(l-partition)
 else
  return l*(t-pos)-logterm
 end
end
