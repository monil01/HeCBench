"""CPU-only exact audit of saved typed-filter captures; never runs GPU kernels."""
from pathlib import Path
import json,struct,hashlib,argparse
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,default=ROOT/'.porting-state/portable_julia/runs/20261018T040059959771Z_convolution1D/result.json');args=parser.parse_args();res=json.loads(args.result.read_text());meta=json.loads((P/'fixture_description.json').read_text());raw=(P/'fixture.bin').read_bytes();words=struct.unpack('<12432Q',raw)
def dec(w,T):return struct.unpack({'Float64':'<d','Float32':'<f','Int16':'<h'}[T],struct.pack({'Float64':'<Q','Float32':'<I','Int16':'<H'}[T],w))[0]
inputs=[[int(dec(w,c['dtype'])) for w in words[c['offset']:c['offset']+2048]] for c in meta['input_cases']];gold=bytearray(raw);guard=struct.pack('<d',-12345.0)*8
for group in meta['groups']:
 a=inputs[group['input_case']];mw=group['mask_width'];mask=[int(dec(w,group['dtype'])) for w in words[group['mask_offset']:group['mask_offset']+mw]];out=[0]*2048
 # Independent scatter formulation rather than the GPU output-centred gather.
 for q,v in enumerate(a):
  for j,c in enumerate(mask):
   i=q+mw//2-j
   if 0<=i<2048:out[i]+=v*c
 payload=struct.pack('<2048d',*map(float,out));gold.extend((guard+payload+guard)*15)
assert len(gold)==755472*8
rows=[];hashes=set()
for model,v in res['models'].items():
 assert len(v['trials'])==5 and all(t['correct'] for t in v['trials']);assert abs(sum(t['time_seconds'] for t in v['trials'][1:])/4-v['mean_seconds'])<1e-14
 for trial in v['trials']:
  data=Path(trial['output_artifact']['saved_path']).read_bytes();assert data==gold,(model,trial['trial']);sha=hashlib.sha256(data).hexdigest();assert sha==trial['output_artifact']['sha256'];hashes.add(sha);rows.append({'model':model,'trial':trial['trial'],'full_exact_pass':True,'input_prefix_bytes':99456,'typed_outputs_checked':737280,'exact_guards_checked':5760,'sha256':sha})
 for source,sha in v['adapter_file_hashes'].items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==sha,(model,source)
print(json.dumps({'benchmark':'convolution1D','CPU_only_no_new_GPU':True,'result':str(args.result),'all20_bitidentical':len(hashes)==1,'means_trials2to5_recomputed':True,'all_actual_dependencies_current_hash_match':True,'independent_expected_full_artifact_sha256':hashlib.sha256(gold).hexdigest(),'oracle':'Int64input-scatterlinearoperator, exactwidenedFloat64IEEEoutputwords andtypedguards; preservedactualtypedIEEEinputprefix','rows':rows},indent=2))
