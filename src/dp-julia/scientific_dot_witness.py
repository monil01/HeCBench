"""CPU-only complete saved-artifact audit; does not execute/rewrite benchmarks."""
from pathlib import Path
import array,hashlib,json,struct,sys
P=Path(__file__).resolve().parent;R=P.parents[1]

def audit(result_path):
 raw=(P/'study_inputs_i32.bin').read_bytes();ints=struct.unpack('<%di'%(len(raw)//4),raw);prefix=[];expected=[]
 for dtype in ['float32','float64']:
  at=0
  for case in range(2):
   n,pad=ints[at:at+2];at+=2;a=ints[at:at+pad];at+=pad;b=ints[at:at+pad];at+=pad
   assert all(x==0 for x in a[n:]+b[n:]);prod=[x*y for x,y in zip(a,b)];assert sum(abs(x)for x in prod)<2**24
   prefix.extend([n,pad,*a,*b]);expected.append(sum(prod))
 encoded=struct.pack('<%dd'%len(prefix),*prefix);assert len(prefix)==90120
 d=json.loads(result_path.read_text());checks=[]
 for model,v in d['models'].items():
  assert len(v['trials'])==5 and all(t['correct']for t in v['trials'])
  for t in v['trials']:
   f=Path(t['output_artifact']['saved_path']);blob=f.read_bytes();assert blob[:len(encoded)]==encoded
   out=struct.unpack('<4d',blob[len(encoded):]);assert list(out)==expected
   checks.append({'model':model,'trial':t['trial'],'sha256':hashlib.sha256(blob).hexdigest(),'artifact':str(f),'input_prefix_values':90120,'actual_dot_outputs':4,'exact_integer_oracle':True})
 return {'run_id':d['run_id'],'configuration_id':d['configuration_id'],'expected_dot_outputs':expected,'all20pass':True,'checks':checks}

if __name__=='__main__':
 records=[audit(Path(p))for p in sys.argv[1:]];print(json.dumps({'records':records,'method':'ActualGPUbufferreadback prefix exact, allfourDOTscalars exactindependentintegerproducts/sums, absbound<2^24 ensuresanyFP32/64ordering. Float64artifactencodinglosslessonthisintegerfixture, notFP64claimforallpaths.'},indent=2))
