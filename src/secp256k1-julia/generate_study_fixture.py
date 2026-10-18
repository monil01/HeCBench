"""Source point tables and independent arbitrary-integer group/field oracle."""
from pathlib import Path
import re,struct,hashlib,json
p=Path(__file__).resolve().parent;src=(p.parent/'secp256k1-cuda/main.cu').read_text();P=2**256-2**32-977;points=[];storage=[]
for m in re.finditer(r'^\s+SC\(([^)]*)\)',src,re.M):
 w=[int(x.strip().removesuffix('u')) for x in m[1].split(',')];assert len(w)==16
 x=sum(v<<(32*(7-i)) for i,v in enumerate(w[:8]));y=sum(v<<(32*(7-i)) for i,v in enumerate(w[8:]));assert (y*y-x*x*x-7)%P==0
 points.append((x,y));storage.append(list(reversed(w[:8]))+list(reversed(w[8:])))
assert len(points)==512 and len(set(points))==512
orders=[list(range(512)),list(reversed(range(512))),list(range(17,512))+list(range(17))];fixture=[3];gold=[];records=[]
for label,order in zip(['canonical','reverse','rotate17'],orders):
 fixture.extend(v for k in order for v in storage[k]);x,y=points[order[0]];z=1;zall=1;hzero=0;curve_ok=True;prefix=[]
 for k in order[1:]:
  bx,by=points[k];zz=z*z%P;u2=bx*zz%P;s2=by*zz*z%P;hh=(u2-x)%P;ii=(s2-y)%P;hzero+=hh==0;assert hh!=0
  h2=hh*hh%P;h3=h2*hh%P;t=x*h2%P;x3=(ii*ii-2*t-h3)%P;y3=((t-x3)*ii-h3*y)%P;z3=z*hh%P
  x,y,z=x3,y3,z3;zall=zall*z%P;curve_ok&=(y*y-x*x*x-7*pow(z,6,P))%P==0
  prefix.extend(v for val in (x,y,z,zall) for v in val.to_bytes(32,'big'))
 assert curve_ok and zall!=0;inv=pow(zall,-1,P);assert inv*zall%P==1;digest=inv.to_bytes(32,'big');gold.extend(digest+bytes([165]*17))
 records.append({'path':label,'point_count':512,'distinct_points':512,'all_on_curve':True,'all511_Jacobian_steps_on_curve':curve_ok,'zero_H_count':hzero,'nonzero_accumulatedZ':True,'inverse_identity':True,'result_hex':digest.hex(),'final_X':hex(x),'final_Y':hex(y),'final_Z':hex(z),'accumulatedZ':hex(zall),'all511_state_trace_sha256':hashlib.sha256(bytes(prefix)).hexdigest()})
assert records[0]['result_hex']=='bbde464b6355ee6de6deba5ae860f8a66524937eee81dde224a0214efd795d09'
fb=struct.pack('<%dI'%len(fixture),*fixture);gb=bytes(gold);(p/'fixture.bin').write_bytes(fb);(p/'expected_bytes.bin').write_bytes(gb)
(p/'fixture_description.json').write_text(json.dumps({'paths':records,'prefix_count':len(fixture),'output_count':len(gold),'fixture_sha256':hashlib.sha256(fb).hexdigest(),'gold_sha256':hashlib.sha256(gb).hexdigest(),'prime':str(P),'artifact':'ActualUInt32sourcepointtables and1header, final32byteoutput+17UInt8guards/path losslesslywidenedUInt32afterD2H. No floatingpointGPUarithmetic.','scope':'Threefull512pointchains/oneGPUthread andsourcefieldinversionoperator; genericexceptionalgroupcases, signatures, keygeneration/security/sidechannelproperties nottested. Nativealgorithm/sourceunchanged. HistoricalCPU-BigIntmathonlyusedasindependentoracle, nottimedGPUwork.'},indent=2)+'\n')
