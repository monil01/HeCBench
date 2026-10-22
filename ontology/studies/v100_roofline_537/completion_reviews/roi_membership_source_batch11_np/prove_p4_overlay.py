import json,gzip,hashlib,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def file(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
rows=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lk:
 fcntl.flock(lk,fcntl.LOCK_SH)
 ledger=json.loads((S/'benchmarks/p4.json').read_bytes())
 for m in ['cuda','cuda_jl','ka','jacc']:
  mp=W/ledger['models'][m]['profile_evidence_manifest'];pp=mp.parent/'plan.json.gz';sp=mp.parent/'source_proof.json.gz';p=json.loads(gzip.decompress(pp.read_bytes()));proof=json.loads(gzip.decompress(sp.read_bytes()));over=[]
  assert proof['source_result_sha256']==p['reference_result_sha256']and proof['source_config_sha256']==p['raw_config_sha256']
  for x in proof['patches']:
   src=Path(x['source']);old=src.read_bytes();new=x['text'].encode();assert sha(old)==x['source_sha256']==p['execution_bindings'][str(src)];assert sha(new)==x['copy_sha256']==p['execution_bindings'][x['copy']]
   private=f'/tmp/hecbench-roofline-v100-20261020/batch028-remaining-paths/p4/{m}/scratch';changes=[]
   if src.name=='study_cuda.cu':
    changes=[('#include "../p4-cuda/main.cu"','#include "/home/imo/HeCBench_julia/src/p4-cuda/main.cu"'),('std::filesystem::path(__FILE__).parent_path()','std::filesystem::path("/home/imo/HeCBench_julia/src/p4-julia")'),('"/home/imo/HeCBench_julia/.porting-state/portable_julia/raw_sidecars/p4"',f'"{private}/profile_outputs/private_capture/p4"')]
   elif src.name=='study_common.jl':
    changes=[('joinpath(@__DIR__,"fixture_bits.bin")','joinpath("/home/imo/HeCBench_julia/src/p4-julia","fixture_bits.bin")'),('joinpath(@__DIR__,"expected_full_outputs.bin")','joinpath("/home/imo/HeCBench_julia/src/p4-julia","expected_full_outputs.bin")'),('"/home/imo/HeCBench_julia/.porting-state/portable_julia/raw_sidecars/p4"',f'"{private}/profile_outputs/private_capture/p4"')]
   else:
    if m=='cuda_jl':changes.append(('include(joinpath(@__DIR__,"main.jl"))','include(joinpath("/home/imo/HeCBench_julia/src/p4-julia/p4-julia-cuda.jl","main.jl"))'))
    changes.append(('include(joinpath(@__DIR__,"..","study_common.jl"))',f'include("{private}/profile_source/study_common.jl")'))
    if m in ['ka','jacc']:changes.append(('include(joinpath(@__DIR__,"..","postprocess_math.jl"))',f'include(joinpath("{src.parent}","..","postprocess_math.jl"))'))
   transformed=old
   for a,b in changes:assert transformed.count(a.encode())==1;transformed=transformed.replace(a.encode(),b.encode())
   assert transformed==new
   inverse=new
   for a,b in reversed(changes):assert inverse.count(b.encode())==1;inverse=inverse.replace(b.encode(),a.encode())
   assert inverse==old
   cp=Path(x['copy']);present=cp.exists();assert not present or cp.read_bytes()==new
   over.append({'original':str(src),'original_sha256':sha(old),'copy':str(cp),'copy_sha256':sha(new),'retained_text':x['text'],'reviewed_exact_changes':[{'before':a,'after':b}for a,b in changes],'forward_and_inverse_byte_equal':True,'copy_current_if_present_matches':True if present else None,'historical_copy_recreated':False})
  rows.append({'benchmark':'p4','model':m,'selected_plan':file(pp),'retained_capture_source_proof':file(sp),'overlays':over,'scope':'Only reviewed literal include/input/output path substitutions; full forward/inverse bytes prove source launch/math/timer text unchanged; no timing equivalence or sampled invocation phase inferred.'})
 (O/'p4_retained_overlay_proof.json.gz').write_bytes(gzip.compress((json.dumps({'reviewed_models':rows},indent=2)+'\n').encode(),mtime=0))
print('P4 all4 exact overlay inverse passed; no historical copies recreated')
