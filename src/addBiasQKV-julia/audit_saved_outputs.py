"""Postverification CPU-only exact byte/source audit; no GPU timing executions."""
from pathlib import Path
import json,hashlib,struct
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'.porting-state/portable_julia';rows=[];proof=[];results={}
for tag,m,h,d,rep in [('original',8192,12,64,1000),('axis',5,3,16,10)]:
 q=json.loads((S/f'addBiasQKV_jacc_{tag}_attempt01.log').read_text().splitlines()[-1]);rp=Path(q['result']);r=json.loads(rp.read_text());assert r['matched_performance_valid'] and r['reference_agreement'];results[tag]=str(rp.relative_to(R));src=(P/f'study_{tag}_src_u8.bin').read_bytes();bias=(P/f'study_{tag}_bias_u16.bin').read_bytes();sc=(P/f'study_{tag}_scales_f32.bin').read_bytes();expected=(P/f'study_{tag}_expected_u8.bin').read_bytes();prefix=src+bias+sc+struct.pack('<4i',m,h,d,rep);goal=prefix+expected+bytes([165])*128+bytes([90])*256
 for model,z in r['models'].items():
  assert z['provenance_valid'] and z['timing_valid'] and len(z['trials'])==5;assert abs(z['mean_seconds']-sum(t['time_seconds']for t in z['trials'][1:])/4)<1e-14
  for name,digest in z['adapter_file_hashes'].items():
   p=Path(name);p=p if p.is_absolute()else R/p;assert p.exists()and hashlib.sha256(p.read_bytes()).hexdigest()==digest,(name,model)
  proof.append({'case':tag,'model':model,'adapter_file_hashes':z['adapter_file_hashes'],'all_files_exact':True,'source_hash':z['source_hash']})
  for t in z['trials']:
   a=t['output_artifact'];p=Path(a['saved_path']);b=p.read_bytes();assert b==goal and hashlib.sha256(b).hexdigest()==a['sha256'];rows.append({'case':tag,'model':model,'trial':t['trial'],'artifact':str(p.relative_to(R)),'sha256':a['sha256'],'checked_input_prefix_bytes':len(prefix),'checked_FP8_output_bytes':len(expected),'checked_guard_bytes':384,'all_exact':True})
(P/'saved_output_audit.json').write_text(json.dumps({'all40_captures_exact':len(rows)==40,'oracle_strength':'Sourcehostreference plus independentfiniteE4M3grid/RNE-BF16operation/layout evaluator; fixture_provenance.json; no fullattentionmodelqualityclaim.','results':results,'artifacts':rows},indent=2)+'\n');(P/'source_integrity_review.json').write_text(json.dumps({'all8_model_case_sources_exact':len(proof)==8,'checks':proof},indent=2)+'\n');print({'all40_exact':len(rows)==40,'all8_sources_exact':len(proof)==8,'results':results})
