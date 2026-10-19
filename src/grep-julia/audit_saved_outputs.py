#!/usr/bin/env python3
from pathlib import Path
import array,json,hashlib,tempfile,shutil,subprocess,sys
P=Path(__file__).resolve().parent;R=P.parent.parent;F=Path(sys.argv[1])if len(sys.argv)>1 else R/'.porting-state/portable_julia/runs/20261019T210048511401Z_grep/result.json';r=json.loads(F.read_text());expected=(P/'oracle_i32.bin').read_bytes();graph=(P/'oracle_graph_i32.bin').read_bytes();hashes=0;records={}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ids={'cuda':'20261019T205149962848Z_grep','cuda_jl':'20261019T205741817784Z_grep','ka':'20261019T205949538259Z_grep','jacc':'20261019T210048511401Z_grep'}
for m,v in r['models'].items():
 assert v['status']=='native_check_pass'and len(v['trials'])==5 and v['provenance_valid']
 for path,digest in v['adapter_file_hashes'].items():assert sha(path)==digest,path;hashes+=1
 for t in v['trials']:
  a=t['output_artifact'];b=Path(a['saved_path']).read_bytes();assert hashlib.sha256(b).hexdigest()==a['sha256'] and b==expected
 log=R/'.porting-state/portable_julia/runs'/ids[m]/f'{m}_run1.log';parts=[];paths=[]
 for line in log.read_text().splitlines():
  if 'actual_untimed_NFA_graph' in line:
   p=Path(line.split('path=')[1].strip());parts.append(p.read_bytes());paths.append({'path':str(p.relative_to(R)),'sha256':sha(p)})
 full=b''.join(parts);assert len(parts)==10 and full==graph
 records[m]={'actual_untimed_graph_patterns':10,'actual_blocks_each_pattern':512,'graph_full_values':len(full)//4,'all_exact_source_semantic_CPU_control':True,'actual_files':paths,'concatenated_sha256':hashlib.sha256(full).hexdigest(),'mean_last4_seconds':v['mean_seconds']}
# Actual zero template bounds/resource diagnostic, not timedoutput.
log=R/'.porting-state/portable_julia/runs'/ids['jacc']/'jacc_run1.log';line=next(l for l in log.read_text().splitlines()if'scratch_initial_final4612values'in l);p=Path(line.split('path=')[1].split(';')[0]);vals=array.array('i');vals.frombytes(p.read_bytes());assert len(vals)==4612 and all(v==0 for v in vals)
with tempfile.TemporaryDirectory(prefix='grep-cpu-cold-')as tmp:
 root=Path(tmp);out=root/'src/grep-julia';out.mkdir(parents=True);d=root/'.porting-state/diagnostics/grep/external_dataset/testcases';d.mkdir(parents=True);shutil.copy2(R/'.porting-state/diagnostics/grep/external_dataset/testcases/romeojuliet.txt',d)
 for n in ['generate_assets.py','generate_graph_oracle.py','study_plan.json']:shutil.copy2(P/n,out)
 for n in ['generate_assets.py','generate_graph_oracle.py']:subprocess.run([sys.executable,str(out/n)],check=True,capture_output=True)
 cold={}
 for f in out.iterdir():
  if f.suffix=='.bin'or f.name in ('study_params.h','study_params.jl','graph_capacity_proof.json'):
   assert f.read_bytes()==(P/f.name).read_bytes(),f.name;cold[f.name]={'bytes':f.stat().st_size,'sha256':sha(f)}
report={'benchmark':'grep','final_result':str(F.relative_to(R)),'result_sha256':sha(F),'all20_primary_full_artifacts_exact':True,'primary_values_each':len(expected)//4,'actual_input_prefix_values_each':r['input_prefix_count'],'source_asset_hash_checks':hashes,'all_actual_untimed_graph_records':records,'independent_regex_oracle':'Pythonre.fullmatch allprimarymaskvalues; graphcontrol source-semantic, notindependentregexalgorithm','cold_CPU_generated_files_exact':cold,'JACC_actual_scratch':{'path':str(p.relative_to(R)),'sha256':sha(p),'values':4612,'allzero':True},'no_GPU_execution_in_audit':True}
(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'primary_files':20,'values_each':len(expected)//4,'graph_files':40,'hashes':hashes,'cold_files':len(cold)}))
