#!/usr/bin/env python3
"""Independently forecast serialized input length from actual declared problem/time values."""
from pathlib import Path
import ctypes,json,struct,math
p=Path(__file__).resolve().parent;lib=ctypes.CDLL('libm.so.6');lib.jn.argtypes=[ctypes.c_int,ctypes.c_double];lib.jn.restype=ctypes.c_double
records=[]
def shape(x,commutator=False):
 stop=2
 while abs(lib.jn(stop,x))>=1e-15:
  stop+=1
  assert stop<1000000
 return (8 if commutator else 10)+9*(stop-2),stop
for name in ['diffusive','localized','small_lattice','general_tail','spin_pairs']:
 f=(p/(name+'_fixture.bin')).open('rb');h=struct.unpack('<12i',f.read(48));n,maxn,ne,nt,nm,rv,v0,vac,msd,spin,ldos,nl=h;vol,emax=struct.unpack('<2d',f.read(16));f.read(ne*8);time=list(struct.unpack('<'+'d'*nt,f.read(nt*8)));base=15+ne+nt+nl+2*n+4*n*maxn+(rv+(nl if ldos else 0))*(3+2*n);extra=0
 for r in range(rv):
  if vac:
   for dt in time[:-1]:extra+=2*shape(dt*emax)[0]
  if msd:
   extra+=shape(time[0]*emax)[0]+shape(time[0]*emax,True)[0]
   for dt in time[1:]:extra+=2*shape(dt*emax)[0]+shape(dt*emax,True)[0]
  if spin:
   for dt in time[:-1]:extra+=shape(dt*emax)[0]
 records.append({'case':name,'prefix_values':base+extra,'problem_initial_state_prefix':base,'actual_Bessel_scalar_prefix':extra,'unique_scaled_times':sorted(set(t*emax for t in time)),'Bessel_first_excluded_orders':{str(x):shape(x)[1]for x in sorted(set(t*emax for t in time))}})
a={'predicted_input_prefix_values':sum(c['prefix_values']for c in records),'cases':records,'scope':'Sourcework-shape andhostlibmcoefficientstop only, noGPUoutput used. Compareactualretainedprefix_length afterpilot; noacceptancerevision.'};(p/'prefix_work_shape_audit.json').write_text(json.dumps(a,indent=2)+'\n');assert a['predicted_input_prefix_values']==769639;print(a)
