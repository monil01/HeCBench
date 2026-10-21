#!/usr/bin/env python3
"""Reviewed compiler-only launcher. Called only later by parent under Landlock."""
import hashlib,json,subprocess,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=Path(sys.argv[1]);expected=sys.argv[2]
if sha(manifest)!=expected:raise SystemExit('Build manifest changed')
x=json.loads(manifest.read_text());scratch=Path(x['scratch']).resolve()
if Path.cwd().resolve()!=scratch or not scratch.is_relative_to('/tmp'):raise SystemExit('Fresh scratch cwd required')
for p,h in x['input_bindings'].items():
 if sha(p)!=h:raise SystemExit('Compiler input changed '+p)
receipt={'kind':'reviewed_native_library_plus_unchanged_CPU_checker_build','commands':x['commands'],'steps':[],'source_bindings':x['input_bindings'],'kernel_math_changed':False,'target_runs':0}
for cmd in x['commands']:
 output=Path(cmd[cmd.index('-o')+1]).resolve();output.relative_to(scratch)
 if output.exists():raise SystemExit('Refuse existing compiler output')
 step={'command':cmd,'output':str(output)};receipt['steps'].append(step)
 result=subprocess.run(cmd,check=False);step['returncode']=result.returncode
 if result.returncode:
  (scratch/'extra_build_outcome.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(result.returncode)
 step['output_sha256']=sha(output)
receipt['outputs']={s['output']:s['output_sha256']for s in receipt['steps']};receipt['checker']=str(scratch/'mtcheck');receipt['checker_sha256']=sha(scratch/'mtcheck')
(scratch/'extra_build_outcome.json').write_text(json.dumps(receipt,indent=2)+'\n')
