from pathlib import Path
import json,hashlib,struct,math
ROOT=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;bundle=ROOT/'.porting-state/portable_julia/runs/20261018T013708669897Z_ssm';d=json.loads((bundle/'result.json').read_text());meta=json.loads((p/'fixture_description.json').read_text());prefix=(p/'fixture.bin').read_bytes();fv=struct.unpack('<'+str(meta['prefix_count'])+'f',prefix);gold=struct.unpack('<'+str(meta['output_count'])+'d',(p/'expected_outputs.bin').read_bytes());frames=[];pos=1
for case in meta['cases']:
 b,dim,L=map(int,fv[pos:pos+3]);pos+=3;arr=[]
 for _ in range(9):n=int(fv[pos]);pos+=1;arr.append(fv[pos:pos+n]);pos+=n
 frames.append((b,dim,L,arr))
assert pos==len(fv)
checks=[];timings={};reset_witness=0.;gpos=0;analytic=[]
for b,dim,L,a in frames:
 u,delta,A,B,C,D,db,z,h0=a;ny=b*dim*L;ns=b*dim*8;g1=gold[gpos:gpos+ny];gpos+=ny+17+ns+17;g3=gold[gpos:gpos+3*(ny+17)+ns+17]
 reset_witness=max(reset_witness,max(abs(g3[ny+17+i]-g1[i]) for i in range(ny)));gpos+=len(g3)
assert reset_witness>1e-5
for model,v in d['models'].items():
 ts=v['trials'];assert len(ts)==5 and all(t['correct'] for t in ts);assert math.isclose(v['mean_seconds'],sum(t['time_seconds'] for t in ts[1:])/4,rel_tol=1e-14)
 timings[model]={'all_five_path_times_seconds':[t['per_path_times_seconds'] for t in ts],'mean_path_times_last4_seconds':[sum(t['per_path_times_seconds'][i] for t in ts[1:])/4 for i in range(6)],'mean_sum_seconds':v['mean_seconds']}
 for t in ts:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(prefix)]==prefix;got=struct.unpack('<'+str(meta['output_count'])+'f',raw[len(prefix):]);ma=max(abs(x-r) for x,r in zip(got,gold));assert ma<=1e-3 and all(math.isfinite(x) for x in got);off=0;guards=0;ae=0.
  for b,dim,L,a in frames:
   u,delta,A,B,C,D,db,z,h0=a;ny=b*dim*L;ns=b*dim*8
   for chain in [1,3]:
    for step in range(chain):
     assert got[off+ny:off+ny+17]==(-12345.,)*17;guards+=17;off+=ny+17
    h=got[off:off+ns];assert got[off+ns:off+ns+17]==(-12345.,)*17;guards+=17;off+=ns+17
    for bi in range(b):
     # Zero-input/zero-A channel preserves every component exactly.
     assert h[bi*dim*8:bi*dim*8+8]==h0[bi*dim*8:bi*dim*8+8]
     if dim>1:
      bd=bi*dim+1;totaldt=math.fsum((x if x>20 else math.log1p(math.exp(x))) for x in [delta[bd*L+j]+db[1] for j in range(L)])*chain
      for n in range(8):ae=max(ae,abs(h[bd*8+n]-h0[bd*8+n]*math.exp(A[8+n]*totaldt)))
  assert off==len(got) and guards==306 and ae<=1e-3
  checks.append({'model':model,'trial':t['trial'],'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'prefix_exact':True,'all_306_guards_exact':True,'max_abs_Float64_error':ma,'zero_input_zero_A_hidden_state_exact':True,'max_homogeneous_decay_abs_error':ae})
rec={'benchmark':'ssm','run':d['run_id'],'artifacts_checked':20,'input_prefix_count':meta['prefix_count'],'input_prefix_bytes':len(prefix),'output_count':len(gold),'guards_per_artifact':306,'Julia_artifacts_bit_identical':len(set(x['sha256'] for x in checks if x['model']!='cuda'))==1,'checks':checks,'per_model_timings':timings,'CPU_counterfactual_reset_each_call_max_output_difference':reset_witness,'scope':'Everyy for every chaininvocation/alltimepositions, finalhiddenstates, exactguards/inputprefix. Intermediatehiddenstates pertime notcaptured. Source serialbase/gate/softplus true only.'};out=p/'independent_artifact_audit.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(rec,indent=2)+'\n');tmp.replace(out);print({'artifacts':20,'julia_exact':rec['Julia_artifacts_bit_identical'],'reset_counterfactual':reset_witness,'means_ms':{m:v['mean_sum_seconds']*1e3 for m,v in timings.items()}})
