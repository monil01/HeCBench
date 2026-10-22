import json,gzip,hashlib,fcntl
from pathlib import Path
O=Path(__file__).resolve().parent;W=Path('/home/imo/HeCBench_julia');h=lambda b:hashlib.sha256(b).hexdigest();p=json.loads((O/'proposed_stage_patches.json').read_bytes());proof=json.loads(gzip.decompress((O/'proofpacket.json.gz').read_bytes()))
def evidence(rel,needles):
 path=str(W/rel);d=proof['sources'][path];b=Path(path).read_bytes();assert h(b)==d['sha256'];enc=d['snippet_encoding'];lines=b.decode(enc).splitlines(keepends=True);ev=[]
 for needle in needles:
  ids=[i for i,l in enumerate(lines)if needle in l];assert ids,(path,needle)
  for i in ids:
   st=sum(len(l.encode(enc))for l in lines[:i]);raw=lines[i].encode(enc);assert b[st:st+len(raw)]==raw;ev.append({'path':path,'sha256':h(b),'snippet_encoding':enc,'line':i+1,'start_byte':st,'end_byte':st+len(raw),'text':lines[i]})
 return ev
observations=[
 {'benchmark':'pitch','observation':'The named benchmark uses real row/slice-pitched CUDA allocations and three Float32 sigmoid steps. Its initialized logical/padding inputs repair original undefined-input reference behavior; this source-only review must preserve that conditional native qualification. Five reset-excluded internal batches provide eight path means. Physical padding guards and Float64 host oracle are checked outside the timer. This does not explain performance differences or certify an untouched-native CLI.', 'evidence':evidence('src/pitch-julia/study_common.jl',['cudaMalloc3D','cudaMallocPitch','for _ in 1:STEPS;launch!','padding+=1']), 'scientific_intent':proof['scientific_intent']['pitch']},
 {'benchmark':'prna','observation':'Five RNA cases time boundary initialization plus eight stages per diagonal in Float32 negative-log partition arithmetic, including impossible +Inf states. Host pair probabilities and a 27-base 209-snapshot replay follow the timed work. The same helpers execute both timed and diagnostic work, so matching a kernel family cannot identify sampled phase. The unused zero temperature field does not imply a 0 K physical simulation; source parameter normalization uses 310.15 K. Numerical equivalence does not establish experimental structure prediction.', 'evidence':evidence('src/prna-julia/study_common.jl',['function launch_pipeline','t=time_ns();launch_pipeline','traceat=0;launch_init','launch_stage!(Val(s)']), 'scientific_intent':proof['scientific_intent']['prna']},
 {'benchmark':'projectile','observation':'The Float32 algebraic map deliberately retains source height=(v*v*sin(angle)^2)/2*9.81, rather than dividing by gravity. Repeated formula evaluation and native agreement therefore do not validate a physical projectile model. Source output/readback checks are outside the launch-loop timer; matching a family does not identify an angle or sampled phase.', 'evidence':evidence('src/projectile-julia/study_common.jl',['h[i]=','t0=time_ns()','CUDA.synchronize();elapsed=']), 'scientific_intent':proof['scientific_intent']['projectile']}
]
(O/'companion_scientific_observations.json').write_text(json.dumps({'scope':'Source scientific distinctions only; no new numerical/runtime phase/whole ROI/causal evidence','observations':observations},indent=2)+'\n')
encoding_path='/home/imo/HeCBench_julia/src/projectile-cuda/Projectile.hpp';encdoc=proof['sources'][encoding_path];assert encdoc['sha256']=='c5ad1618143549433ec76937c5e21beb06dcc0af0ec046c0dd3b2e02fd9352bf'and encdoc['snippet_encoding']=='latin-1';b=Path(encoding_path).read_bytes();assert b==encdoc['snippet'].encode('latin-1');snippetchecks=[]
for row in p['patches']:
 for st in row['stage_patches']:
  pr=st['updates']['ROI_membership_proof']
  for e in pr['evidence']+pr['timed_closure_boundaries']:
   if e['path']!=encoding_path:continue
   ls=b.decode('latin-1').splitlines(keepends=True)
   for q in e['line_evidence']:
    i=q['line']-1;raw=ls[i].encode('latin-1');start=sum(len(x.encode('latin-1'))for x in ls[:i]);assert ls[i].rstrip('\r\n')==q['text'];assert b[start:start+len(raw)]==raw;snippetchecks.append({'benchmark':row['record_key']['benchmark'],'model':row['record_key']['model'],'stage_id':st['stage_id'],'line':i+1,'start_byte':start,'end_byte':start+len(raw),'snippet_encoding':'latin-1','snippet':ls[i]})
(O/'reversible_encoding_proof.json').write_text(json.dumps({'status':'PRIVATE_ROOT_APPROVAL_REQUIRED','proposed_approved_reversible_source_encodings':{encoding_path:{'sha256':encdoc['sha256'],'encoding':'latin-1'}},'full_source':encdoc,'every_selected_line_snippet_byte_proofs':snippetchecks,'non_utf8_byte_offsets':[i for i,x in enumerate(b)if x>=128],'byte_equality_roundtrip':True,'generic_fallback':False,'replace_or_ignore':False,'scope':'Only this exact path/hash. Root must approve exact mapping in root note before merge; no source bytes changed.'},indent=2)+'\n')
quotechecks=0
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for d in proof['sources'].values():
  bb=Path(d['path']).read_bytes();enc=d['snippet_encoding'];assert h(bb)==d['sha256'];assert bb[d['start_byte']:d['end_byte']]==d['snippet'].encode(enc);assert d['end_line']==bb.decode(enc).count('\n')+1;quotechecks+=1
 for row in p['patches']:
  for patch in row['stage_patches']:
   d=patch['updates']['ROI_membership_proof']['exact_stage_callsite'];path=str(Path(d['path']).resolve());bb=Path(path).read_bytes();assert h(bb)==d['source_sha256'];enc=proof['sources'][path]['snippet_encoding'];lines=bb.decode(enc).splitlines();ctx=d['context'].splitlines();start=d['context_start_line']-1;assert lines[start:start+len(ctx)]==ctx;quotechecks+=1
 for d in proof['scientific_intent'].values():assert h(Path(d['binding']['path']).read_bytes())==d['binding']['sha256'];quotechecks+=1
 (O/'source_quote_revalidation.json').write_text(json.dumps({'exact_fullsource_bytes_stage_contexts_and_domain_hashes_checked':quotechecks,'all_equal':True,'metadata_lock':'LOCK_SH','source_only':True,'exact_latin1_header_fulltext_and_all_line_snippets_reversible':True},indent=2)+'\n')
matchrows=[x for r in p['patches']for x in r['selected_observation_matching']];counts={'models':len(p['patches']),'names':len({x['record_key']['benchmark']for x in p['patches']}),'updates':sum(len(x['stage_patches'])for x in p['patches']),'inclusions':sum(s['updates']['semantic_family_in_primary_ROI']for x in p['patches']for s in x['stage_patches']),'saved_rows':len(matchrows),'matched_saved_rows':sum(x['source_family_inclusion_only']for x in matchrows),'unassigned_saved_rows':sum(not x['source_family_inclusion_only']for x in matchrows),'inclusion_alias_nulls':sum(s['updates']['semantic_family_in_primary_ROI']and s['updates'].get('model_kernel_regex','MISSING')is None for x in p['patches']for s in x['stage_patches'])};(O/'review_counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(counts,'QUOTE_CHECKS',quotechecks,'LATIN_SNIPPETS',len(snippetchecks))
