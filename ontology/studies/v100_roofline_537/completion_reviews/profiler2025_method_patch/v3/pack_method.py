"""ROOT-ONLY prospective packer; CPU saved-data validation, no process execution.
Author did not run this packer. Requires a completed calibration and root note.
"""
import argparse,base64,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'proposed/tools'))
import v100_profiler_method as gate

def main():
 p=argparse.ArgumentParser();p.add_argument('--calibration-root',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--root-review-note',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
 root=a.calibration_root.resolve();out=a.output_root.resolve()
 if out.exists():raise ValueError('Fresh output directory required')
 note=a.root_review_note.read_bytes();gate.need(len(note.strip())>=32,'root scientific review note required')
 manifest=a.source_manifest.read_bytes();gate.checked(manifest,gate.MANIFEST);m=json.loads(manifest);prov=(root/'profile_provenance.json').read_bytes();pr=json.loads(prov)
 gate.need(pr.get('status')=='new_profiler_calibration_pass_root_adoption_still_required','calibration pending/failed')
 f={'manifest.json':manifest,'provenance.json':prov,'controller.py':(root/'controller_source.py').read_bytes(),'parser.py':(root/'parser_source.py').read_bytes(),'root_review_note.txt':note,'precision/version.log':(root/'precision/version.log').read_bytes(),'precision/version.payload.bin':(root/'precision/version.payload.bin').read_bytes()}
 for case,spec in m['cases'].items():
  for name in ['counters.csv','execution.log','identity.log','identity.payload.bin']:f[case+'/'+name]=(root/case/name).read_bytes()
  f[case+'/old_counters.csv']=Path(spec['old_counters']).read_bytes();f[case+'/source.cu']=Path(spec['source']).read_bytes()
  for name in ['old_execution_log','old_provenance']:f[case+'/'+name+'.txt']=Path(spec[name]).read_bytes()
 packed={'schema_version':1,'files':{k:{'sha256':gate.sha(v),'base64':base64.b64encode(v).decode()}for k,v in f.items()}}
 bundle=(json.dumps(packed,sort_keys=True,separators=(',',':'))+'\n').encode()
 normalized=gate.normalize_version_log(f['precision/version.log'],[str(root)])
 ident={'version_stdout_sha256':normalized['payload_sha256'],'retained_version_log_sha256':normalized['merged_log_sha256'],'executable':m['ncu_executable'],'executable_sha256':m['bindings'][m['ncu_executable']],'target_ELF':m['ncu_target_ELF'],'target_ELF_sha256':m['bindings'][m['ncu_target_ELF']],'version_stdout':normalized['text']}
 review={'schema_version':1,'method_id':gate.METHOD,'status':'root_approved_after_saved_calibration_pass','profiler':ident,'bundle_file':'method_bundle.json','bundle_sha256':gate.sha(bundle),'root_review_note_sha256':gate.sha(note),'timing_comparability':'Separate profiler method; primary times unchanged; no performance-equivalence claim'}
 raw=(json.dumps(review,indent=2)+'\n').encode();gate.validate_saved(raw,gate.sha(raw),bundle)
 out.mkdir(parents=True);(out/'method_bundle.json').write_bytes(bundle);(out/'review.json').write_bytes(raw);print(json.dumps({'review':str(out/'review.json'),'sha256':gate.sha(raw),'bundle_sha256':gate.sha(bundle)}))
if __name__=='__main__':main()
