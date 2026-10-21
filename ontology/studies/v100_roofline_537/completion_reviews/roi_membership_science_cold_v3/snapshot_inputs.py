"""Short read-only metadata LOCK_SH snapshot, no GPU lock or target execution."""
import fcntl,pathlib,json,gzip,hashlib,datetime,argparse
P=pathlib.Path;W=P('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=P(__file__).parent;N=('axhelm','bicgstab','ccsd-trpdrv','clenergy','chemv');MODELS=('cuda','cuda_jl','ka','jacc')
def sha(b):return hashlib.sha256(b).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--label',required=True);arg=a.parse_args();dest=O/arg.label
if dest.exists():raise ValueError('Fresh snapshot label required')
dest.mkdir();files={};selections={}
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 began=datetime.datetime.now(datetime.timezone.utc).isoformat()
 def capture(p):
  b=p.read_bytes();rel=p.relative_to(W);d=dest/rel;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(b);files[str(p)]={'path':str(p),'sha256':sha(b),'snapshot':str(d),'bytes':len(b)}
  if p.suffix=='.gz':files[str(p)]['uncompressed_sha256']=sha(gzip.decompress(b))
  return b
 for name in('batch003_names1_80.json.gz','batch003_names81_160_review.json.gz'):capture(S/'roi_reviews'/name)
 for n in N:
  path=S/'benchmarks'/(n+'.json');d=json.loads(capture(path))
  for m in MODELS:
   x=d['models'][m]
   if x['profiling_status']!='profile_process_completed'or not x.get('kernel_observations'):raise ValueError('Stable selected completion unavailable '+n+'/'+m)
   obs=W/x['kernel_observations']['path'];folder=obs.parent
   for name in('manifest.json','plan.json.gz','profile_provenance.json.gz','derived.json.gz'):capture(folder/name)
   if str(obs)not in files:capture(obs)
   selections[n+'/'+m]={'benchmark':n,'model':m,'ledger':str(path),'kernel_observations':x['kernel_observations'],'profile_ids':x['profile_ids'],'folder':str(folder),'profiling_status':x['profiling_status']}
 ended=datetime.datetime.now(datetime.timezone.utc).isoformat();fcntl.flock(lock,fcntl.LOCK_UN)
report={'lock':'/tmp/hecbench-roofline-metadata.lock','mode':'existingreadonlyLOCK_SH','started_at':began,'finished_at':ended,'files':files,'selected':selections};(dest/'snapshot.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'label':arg.label,'files':len(files),'models':len(selections),'start':began,'end':ended}))
