"""Independent saved-byte/Float64-oracle audit, no GPU or timing changes."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
p=Path(__file__).resolve().parent;prefix=(p/'study_input_f32.bin').read_bytes();expected=(p/'study_expected_u8.bin').read_bytes();boxes=np.frombuffer(prefix,dtype='<f4').reshape(4096,4);assert np.isfinite(boxes).all();bitmap=np.frombuffer(expected[:4096*4096],dtype=np.uint8).reshape(4096,4096);keep=np.frombuffer(expected[4096*4096:],dtype=np.uint8);assert np.all(keep==np.all(bitmap!=0,axis=1));assert np.all((bitmap==0)|(bitmap==1));assert int(keep.sum())==1024;rows={}
for arg in sys.argv[1:]:
 rp=Path(arg);r=json.loads(rp.read_text());models={}
 for m,x in r['models'].items():
  trials=[]
  for t in x['trials']:
   f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert raw==prefix+expected
   trials.append({'trial':t['trial'],'artifact':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'input_prefix_byteexact':True,'all_pair_bitmap_exact':True,'all_keep_flags_exact':True,'max_abs_error':0,'max_rel_error':0})
  models[m]=trials
 rows[r['configuration_id']]={'result':str(rp),'models':models}
old=(p/'reference_original/main.jl').read_text();current=(p/'nms-julia-cuda.jl/main.jl').read_text();section=lambda s:s[s.index('function generate_nms'):s.index('function resolve_input')];assert section(old)==section(current)
out={'benchmark':'nms','all_input_values_finite':True,'source_threshold_margin':json.loads((p/'fixture_provenance.json').read_text())['minimum_active_threshold_margin'],'kept':1024,'pairwise_values':4096*4096,'keep_values':4096,'actual_input_prefix_bytes':65536,'method':'All40savedartifactsbyteexact versusfullFloat64predicateoracle+rowAND; originalCUDA.jlkernelbodiesbytepreserved. Noactualdetectoraccuracyclaim.','existing_cuda_jl_kernel_bodies_unchanged':True,'configurations':rows};(p/'saved_artifact_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('40fullartifacts exact, allinputfloatfinite, originalJuliaGPUbodiespreserved; keep1024.')
