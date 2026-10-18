"""Fresh untimed corroboration, not reconstruction of deleted primary child logs."""
import pathlib,json,hashlib,subprocess,os,datetime
P=pathlib.Path(__file__).resolve().parent;R=P.parent.parent
S=R/'.porting-state/portable_julia';D=S/'diagnostics/assert/supplemental';D.mkdir(parents=True,exist_ok=True)
c=json.load(open(P/'study_config.json'));rows=[]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for model,spec in c['models'].items():
 command=spec['command'][2:]
 if model=='cuda':command=['/tmp/assert_supplemental_native']
 for t in range(1,6):
  started=datetime.datetime.now(datetime.timezone.utc).isoformat()
  child=subprocess.run(command+['--negative'],cwd=spec['cwd'],capture_output=True,text=True)
  stdout=D/f'{model}_negative_{t}.stdout.log';stderr=D/f'{model}_negative_{t}.stderr.log';stdout.write_text(child.stdout);stderr.write_text(child.stderr)
  statuses=[int(x.split()[1]) for x in child.stdout.splitlines() if x.startswith('ASSERT_STATUS ')]
  row=dict(model=model,diagnostic=t,started_at=started,command=command+['--negative'],cwd=spec['cwd'],returncode=child.returncode,actual_statuses=statuses,stdout={'path':str(stdout.relative_to(R)),'sha256':sha(stdout)},stderr={'path':str(stderr.relative_to(R)),'sha256':sha(stderr)},CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),timing_accepted=False,numerical_agreement_claim=False,fresh_corroboration_not_original_execution=True)
  rows.append(row);(D/'observations.json').write_text(json.dumps(rows,indent=2)+'\n');print(model,t,child.returncode,statuses,flush=True)
  assert child.returncode==0 and statuses==[710]
print('ALL20_FRESH_ISOLATED_ACTUAL710_DIAGNOSTICS_PASS')
