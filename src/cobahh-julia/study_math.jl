@inline timestep(t,dt)=trunc(Int32,(t+1f-3*dt)/dt)
@inline function point!(i,d_h,d_m,d_n,d_ge,d_v,d_gi,d_lastspike,d_not_refractory,_N,dt,t,P)
 if i<=_N
  _lio_1=Int32(P[1])
  _lio_2=P[2]
  _lio_3=P[3]
  _lio_4=P[4]
  _lio_5=P[5]
  _lio_6=P[6]
  _lio_7=P[7]
  _lio_8=P[8]
  _lio_9=P[9]
  _lio_10=P[10]
  _lio_11=P[11]
  _lio_12=P[12]
  _lio_13=P[13]
  _lio_14=P[14]
  _lio_15=P[15]
  _lio_16=P[16]
  _lio_17=P[17]
  _lio_18=P[18]
  _lio_19=P[19]
  _lio_20=P[20]
  _lio_21=P[21]
  _lio_22=P[22]
  _lio_23=P[23]
  _lio_24=P[24]
  _lio_25=P[25]
  _lio_26=P[26]
  _lio_27=P[27]
  _lio_28=P[28]
  _lio_29=P[29]
  _lio_30=P[30]
  _lio_31=P[31]
  _lio_32=P[32]
  _lio_33=P[33]
 @inbounds begin
  h = d_h[i]
  m = d_m[i]
  n = d_n[i]
  ge = d_ge[i]
  v = d_v[i]
  lastspike = d_lastspike[i]
  gi = d_gi[i]
  not_refractory = timestep(t - lastspike, dt) >= _lio_1
  _BA_h = (_lio_2 * exp(_lio_3 * v))/(((-4.0f0)/(0.001f0 + (_lio_4 * exp(_lio_5 * v)))) - (_lio_2 * exp(_lio_3 * v)))
  _h = (- _BA_h) + ((_BA_h + h) * exp(dt * (((-4.0f0)/(0.001f0 + (_lio_4 * exp(_lio_5 * v)))) - (_lio_2 * exp(_lio_3 * v)))))
  _BA_m = (((_lio_6/(_lio_7 + (_lio_8 * exp(_lio_9 * v)))) + (_lio_10/(_lio_7 + (_lio_8 * exp(_lio_9 * v))))) - ((0.32f0 * v)/(_lio_7 + (_lio_8 * exp(_lio_9 * v)))))/(((((_lio_11/(_lio_7 + (_lio_8 * exp(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * exp(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * exp(_lio_15 * v))))) + ((0.32f0 * v)/(_lio_7 + (_lio_8 * exp(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * exp(_lio_9 * v)))) + ((0.28f0 * v)/(_lio_13 + (_lio_14 * exp(_lio_15 * v))))))
  _m = (- _BA_m) + ((_BA_m + m) * exp(dt * (((((_lio_11/(_lio_7 + (_lio_8 * exp(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * exp(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * exp(_lio_15 * v))))) + ((0.32f0 * v)/(_lio_7 + (_lio_8 * exp(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * exp(_lio_9 * v)))) + ((0.28f0 * v)/(_lio_13 + (_lio_14 * exp(_lio_15 * v))))))))
  _BA_n = (((_lio_17/(_lio_7 + (_lio_18 * exp(_lio_5 * v)))) + (_lio_19/(_lio_7 + (_lio_18 * exp(_lio_5 * v))))) - ((0.032f0 * v)/(_lio_7 + (_lio_18 * exp(_lio_5 * v)))))/(((_lio_20/(_lio_7 + (_lio_18 * exp(_lio_5 * v)))) + ((0.032f0 * v)/(_lio_7 + (_lio_18 * exp(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * exp(_lio_5 * v)))) + (_lio_21 * exp(_lio_22 * v))))
  _n = (- _BA_n) + ((_BA_n + n) * exp(dt * (((_lio_20/(_lio_7 + (_lio_18 * exp(_lio_5 * v)))) + ((0.032f0 * v)/(_lio_7 + (_lio_18 * exp(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * exp(_lio_5 * v)))) + (_lio_21 * exp(_lio_22 * v))))))
  _ge = _lio_23 * ge
  _BA_v = (_lio_24 + ((((_lio_25 * (n*n*n*n)) + (_lio_26 * (h * (m*m*m)))) + (_lio_27 * ge)) + (_lio_28 * gi)))/((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)))
  _v = (- _BA_v) + ((_BA_v + v) * exp(dt * ((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)))))
  _gi = _lio_33 * gi
  
  d_h[i] = _h
  d_m[i] = _m
  d_n[i] = _n
  d_ge[i] = _ge
  d_v[i] = _v
  d_gi[i] = _gi
  d_not_refractory[i] = not_refractory
 end
 end
 return
end
