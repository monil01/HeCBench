"""Analytic photon source/Compton/Rayleigh controls, not a second transport solver."""
from pathlib import Path
import argparse,struct,json,math,hashlib
p=argparse.ArgumentParser();p.add_argument('--capture-dir',type=Path,required=True);p.add_argument('--rng-capture',type=Path,required=True);p.add_argument('--input-capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();D=a.capture_dir;P=a.input_capture
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
read=lambda path,fmt:list(struct.unpack('<'+fmt*(path.stat().st_size//struct.calcsize(fmt)),path.read_bytes()))
rng=read(a.rng_capture,'f');edges=read(P/'initial_source_edges.bin','f');weights=read(P/'initial_source_weights.bin','f');aliases=read(P/'initial_source_alias.bin','i');assert len(rng)==384
# Engineering controls declared before evaluating captured values.
ABS=1e-7;REL=2e-6;KNORM=1e-3;KCOMPTON=1e-5
records=[];max_source_normalized=0.;max_compton=0.;max_norm=0.
for kind in range(3):
 raw=(D/f'identity{kind}.bin').read_bytes();assert len(raw)==44*32
 for lane in range(32):
  row=struct.unpack_from('<BBbBffII6f',raw,44*lane);status,reserved,charge,process,e,wt,region,latch,x,y,z,u,v,w=row
  assert status==2 and reserved==0 and charge==0 and process==1 and all(math.isfinite(q)for q in row[4:]);normerr=abs(u*u+v*v+w*w-1);assert normerr<=KNORM;max_norm=max(max_norm,normerr)
  if kind==0:
   assert region==0 and latch==0;assert [x,y,z]==[0,0,-30]
   r1,r2,r3,r4=[rng[k*32+lane]for k in range(4)];aj=f32(r3*145);j=int(aj);fraction=f32(aj-j)
   if fraction>weights[j]:j=aliases[j]
   ee=edges[j]+(edges[j+1]-edges[j])*r4
   dx=f32(-6.4)+f32(12.8)*r1;dy=f32(-6.4)+f32(12.8)*r2;dz=f32(-6.4)+30;length=math.sqrt(dx*dx+dy*dy+dz*dz);expected=[ee,dx/length,dy/length,dz/length,f32(f32(12.8)*f32(12.8))*dz/(length**3)]
   for got,target in zip([e,u,v,w,wt],expected):
    error=abs(got-target)/(ABS+REL*abs(target));assert error<=1;max_source_normalized=max(max_source_normalized,error)
  elif kind==1:
   assert region==1 and latch==1 and 0<e<=f32(.05) and wt==1
   residual=abs((f32(.05)/e-1)-(f32(.05)/f32(.5110034))*(1-w));assert residual<=KCOMPTON;max_compton=max(max_compton,residual)
  else:
   assert region==1 and latch==65536 and e==f32(.05) and wt==1
 records.append({'interaction':['source','Compton','Rayleigh'][kind],'all32_values_pass':True,'capture_sha256':hashlib.sha256(raw).hexdigest()})
report={'all_pass':True,'controls':records,'source_values_compared':160,'source_max_normalized_engineering_error':max_source_normalized,'source_budget':{'absolute':ABS,'relative':REL},'max_Compton_dimensionless_energy_angle_residual':max_compton,'Compton_residual_budget':KCOMPTON,'max_squared_direction_norm_error':max_norm,'direction_budget':KNORM,'scope':'Originalhelperfunctions execute complete32-lane warp usingrealcorrected inputtables andsourceMTGP. IndependentF64analytic solid-angleemission/aliasenergy bounds, free-electronCompton energy-angleidentity, Rayleighelasticenergy andunitdirections. Notfulltransportorstatisticalcrosssectionaccuracy certificate.','input_manifest_sha256':hashlib.sha256((P/'capture_manifest.json').read_bytes()).hexdigest(),'rng_capture_sha256':hashlib.sha256(a.rng_capture.read_bytes()).hexdigest()};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
