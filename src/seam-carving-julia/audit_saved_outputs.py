"""CPU-only postverification audit; never creates a GPU timing series."""
from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'.porting-state/portable_julia';results={};rows=[];proof=[]
input=np.fromfile(P/'study_pixels_u32.bin','<u4').reshape(128,128)
for mode in range(3):
 log=S/f'seam-carving_jacc_mode{mode}_attempt01.log';rp=Path(json.loads(log.read_text().splitlines()[-1])['result']);r=json.loads(rp.read_text());assert r['matched_performance_valid'] and r['reference_agreement'];results[str(mode)]=str(rp.relative_to(R));expected=np.fromfile(P/f'study_expected_mode{mode}_u32.bin','<u4')
 for model,z in r['models'].items():
  assert z['provenance_valid'] and len(z['trials'])==5 and z['timing_valid']
  assert abs(z['mean_seconds']-sum(t['time_seconds'] for t in z['trials'][1:])/4)<1e-14
  for f,h in z.get('adapter_file_hashes',{}).items():
   p=Path(f);p=p if p.is_absolute()else R/p
   assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==h,(model,p)
  proof.append({'mode':mode,'model':model,'adapter_files_exact':True,'adapter_file_hashes':z.get('adapter_file_hashes',{}),'source_hash':z['source_hash']})
  for t in z['trials']:
   a=t['output_artifact'];p=Path(a['saved_path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==a['sha256'];out=np.frombuffer(b,'<u4');assert np.array_equal(out,expected);assert np.array_equal(out[:16384],input.ravel());assert np.array_equal(out[16384:16391],np.array([128,128,100,mode,1,29,28],np.uint32));img=out[16391:16391+128*28].reshape(128,28)
   # Independent property: every output row is an ordered RGB subsequence.
   for before,after in zip(input,img):
    j=0
    for pix in before:
     if j<len(after) and pix==after[j]:j+=1
    assert j==len(after)
   seam=out[16391+128*28:16391+128*28+128].astype(np.int32);assert np.all((seam>=0)&(seam<29)) and np.all(abs(np.diff(seam))<=1)
   rows.append({'mode':mode,'model':model,'trial':t['trial'],'artifact':str(p.relative_to(R)),'sha256':a['sha256'],'exact_input_prefix':True,'exact_full_source_behavior_oracle':True,'independent_ordered_RGB_row_subsequence':True,'last_seam_valid_and_adjacent':True,'checked_payload_values':len(out)-16391})
summary={'all60_artifacts_exact':len(rows)==60,'all5_dual_reference_checks_each_config':True,'source_backtracking_quality_separate':'diagnostics/native_quality_witness.json','CPU_oracle_strength':'Exact source-output emulator plus independent row-subsequence and path-validity properties. Minimum-forward-energy quality evaluated separately on actual native preimages/traces, not inferred from translation PASS.','numpy':np.__version__,'results':results,'artifacts':rows}
(P/'saved_output_audit.json').write_text(json.dumps(summary,indent=2)+'\n');(P/'source_integrity_review.json').write_text(json.dumps({'all12_model_configuration_dependencies_exact':len(proof)==12,'checks':proof},indent=2)+'\n');print({'all60_artifacts_exact':len(rows)==60,'source_configurations':len(proof),'results':results})
