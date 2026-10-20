"""Recorded one-time coordinator operation, not a resume entrypoint.
Do not rerun after source receipts exist; resume from the joint checkpoint.
"""
import json,pathlib,hashlib,subprocess
r=pathlib.Path('/home/imo/HeCBench_julia');p=r/'src/word2vec-julia/async_workflow';ready=json.loads((p/'ready_support_files.json').read_text());archive=r/'src/word2vec-julia/reference_history/before_async_workflow_commits_20261020';archive.mkdir(parents=True,exist_ok=True);original={}
for name,sha in ready['file_hashes'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha,name
print('All',len(ready['file_hashes']),'word2vec packet file hashes match',flush=True)
for model,suffix in [('cuda_jl','cuda.jl'),('ka','KA'),('jacc','JACC')]:
 files=[r/f'.porting-state/porting_logs/word2vec-julia-{model}/manifest.json',r/f'.porting-state/portable_julia/status_300/word2vec/{model}.json',r/f'src/word2vec-julia/word2vec-julia-{suffix}/verification.json']
 for f in files:
  if not f.exists():continue
  original[str(f.relative_to(r))]=json.loads(f.read_text());dest=archive/(model+'_'+f.parent.name+'_'+f.name);dest.write_bytes(f.read_bytes())
record=r/'src/_portable_study/completion537_20261020/word2vec_original_metadata.json';record.write_text(json.dumps(original,indent=2)+'\n')
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=r,text=True).strip()
paths=ready['shared_paths']+[str(f.relative_to(r))for f in archive.iterdir()]+[str(record.relative_to(r)),str((p/'ready_support_files.json').relative_to(r))]
for name in paths:assert (r/name).stat().st_size<8*1024*1024,name
subprocess.run(['git','add','-f','--',*paths],cwd=r,check=True)
subprocess.run(['git','commit','-m','Record full asynchronous word2vec property workflow and preserved component history','-m','Additional bounded corpus workflow: original20-thread15-epoch controller, fixed scientific property gates, full distinct model/traces, five checked trials; source CPU host and queue interoperation explicit. Frozen numerical component and its effort retained. Earlier synchronization-intervened series preserved and excluded. No raw embedding identity or text8 production-quality claim.'],cwd=r,check=True,stdout=subprocess.DEVNULL)
commits=[{'phase':'workflow_shared','returncode':0,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()}]
for phase in ['shared','KA','JACC']:
 cmd=['python3','tools/commit_verified_portable_variant.py','--benchmark','word2vec','--result','.porting-state/portable_julia/runs/20261020T155223860367Z_word2vec/result.json']+(['--shared']if phase=='shared'else['--variant',phase]);log=pathlib.Path('/tmp')/f'word2vec_workflow_commit_{phase}.log'
 with log.open('w')as stream:q=subprocess.run(cmd,cwd=r,stdout=stream,stderr=subprocess.STDOUT)
 row={'phase':'CUDAjl'if phase=='shared'else phase,'returncode':q.returncode,'log':str(log)}
 if q.returncode==0:row['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
 commits.append(row);print(json.dumps(row),flush=True)
 (r/'src/_portable_study/completion537_20261020/word2vec_workflow_commits.json').write_text(json.dumps({'benchmark':'word2vec','configuration_id':'word2vec_named_async_corpus20threads15epochs_dim128_property_v1','role':'workflow_property_secondary','original_metadata':str(record.relative_to(r)),'commits':commits},indent=2)+'\n')
 if q.returncode:raise SystemExit(q.returncode)
