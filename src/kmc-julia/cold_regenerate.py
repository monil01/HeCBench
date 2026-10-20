#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,tempfile,shutil,json,hashlib,sys
import numpy as np
P=Path(__file__).resolve().parent;a=argparse.ArgumentParser();a.add_argument('--dataset',type=Path,required=True);a.add_argument('--materialize',action='store_true');args=a.parse_args();assert np.__version__=='2.5.4'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.load(open(P/'asset_manifest.json'))['assets']if(P/'asset_manifest.json').exists()else{}
with tempfile.TemporaryDirectory(prefix='kmc-cold-')as td:
 out=Path(td);shutil.copy2(P/'generate_assets.py',out/'generate_assets.py');subprocess.run([sys.executable,str(out/'generate_assets.py'),'--dataset',str(args.dataset.resolve())],check=True);checks=[]
 for f in sorted(out.glob('*.bin'))+list(out.glob('study_plan.json')):
  old=P/f.name;expected=sha(old)if old.exists()else manifest[f.name]['sha256'];checks.append({'file':f.name,'sha256':sha(f),'expected_sha256':expected,'bytes':f.stat().st_size,'same':sha(f)==expected})
 assert all(c['same']for c in checks),checks
 if args.materialize:
  for f in out.glob('*.bin'):
   if not(P/f.name).exists():shutil.copy2(f,P/f.name)
  for child in P.glob('kmc-julia-*'):
   if child.is_dir():
    for f in out.glob('*.bin'):
     if not(child/f.name).exists():(child/f.name).symlink_to(Path('..')/f.name)
 (P/'cold_regeneration_audit.json').write_text(json.dumps({'benchmark':'kmc','status':'pass','CPU_only':True,'numpy':np.__version__,'checks':checks,'command':sys.argv,'CPU_BLAS_architecture_reproducibility_scope':'ExactlocalpinnedNumPyCPUplatformreproduction; differentCPU_BLASreduction kernelsmaychangedoraclelowbits/budgets shouldremainmathematicallyvalid.'},indent=2)+'\n');print('COLD PASS',len(checks),'assets')
