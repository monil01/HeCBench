#!/usr/bin/env python3
"""Cold CPU regeneration in isolated temporary source tree; never changes measured assets.
Requires exact declared DVC input directory and NumPy2.5.4. No GPU calls.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,tempfile
import numpy as np
p=Path(__file__).resolve().parent;r=p.parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--materialize',action='store_true',help='Copy missing regenerated CPU binaries into the source study directory after exact-hash checks');args=ap.parse_args();assert np.__version__=='2.5.4',np.__version__
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
with tempfile.TemporaryDirectory(prefix='heartwall-cold-') as td:
 root=Path(td);out=root/'src/heartwall-julia';out.mkdir(parents=True);(root/'src/heartwall-cuda').symlink_to(r/'src/heartwall-cuda',target_is_directory=True)
 for f in p.iterdir():
  if f.is_file()and f.suffix in ('.py','.cpp','.h','.cu','.inc','.jl','.json'):shutil.copy2(f,out/f.name)
 exe=root/'prepare_assets';cmd=['g++','-O3','-std=c++17','-fpermissive','-x','c++',str(out/'prepare_assets.cpp')]+[str(r/'src/heartwall-cuda'/x)for x in ['util/avi/avilib.c','util/avi/avimod.c','util/file/file.c']]+['-o',str(exe)]
 subprocess.run(cmd,check=True);subprocess.run([str(exe),str(args.data_dir.resolve()),str(out)],check=True)
 for script in ['generate_scientific_oracle.py','generate_final_operator_oracle.py','prepare_layout.py','generate_host_sources.py']:subprocess.run([sys.executable,str(out/script)],check=True)
 checks=[]
 manifest=json.load(open(p/'asset_manifest.json'))['assets'] if (p/'asset_manifest.json').exists() else {}
 def expected(path):return sha(path) if path.exists() else manifest[path.name]['sha256']
 for f in sorted(out.glob('*.bin')):
  old=p/f.name;checks.append({'file':f.name,'sha256':sha(f),'original_sha256':expected(old),'bytes':f.stat().st_size,'same':sha(f)==expected(old)})
 for name in ['study_checker.cpp','host_specs.jl','study_params.h','study_params.jl','buffer_layout.json','artifact_layout.json','scientific_oracle_plan.json']:
  checks.append({'file':name,'sha256':sha(out/name),'original_sha256':sha(p/name),'same':sha(out/name)==sha(p/name)})
 report={'benchmark':'heartwall','status':'pass'if all(x['same']for x in checks)else'fail','CPU_only':True,'numpy':np.__version__,'python':sys.version,'data_dir':str(args.data_dir),'input_hashes':{f.name:sha(f)for f in args.data_dir.iterdir()if f.is_file()},'checks':checks,'regenerated_binary_bytes':sum(x.get('bytes',0)for x in checks),'command':sys.argv}
 (p/'cold_regeneration_audit.json').write_text(json.dumps(report,indent=2)+'\n');assert report['status']=='pass',[(x['file'],x['same'])for x in checks]
 if args.materialize:
  for f in out.glob('*.bin'):
   if not (p/f.name).exists():shutil.copy2(f,p/f.name)
  for child in p.glob('heartwall-julia-*'):
   if child.is_dir():
    for f in out.glob('*.bin'):
     link=child/f.name
     if not link.exists():link.symlink_to(Path('..')/f.name)
 print('COLD PASS',len(checks),'assets')
