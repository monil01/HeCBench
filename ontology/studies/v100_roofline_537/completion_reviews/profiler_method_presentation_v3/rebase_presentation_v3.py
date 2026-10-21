from pathlib import Path
import subprocess,json,hashlib,datetime,difflib
R=Path('/home/imo/HeCBench_julia'); P=Path('/tmp/roofline-completion-20261021/profiler_method_presentation_v2'); O=Path('/tmp/roofline-completion-20261021/profiler_method_presentation_v3')
O.mkdir(exist_ok=False)
files=list(json.loads((P/'authoring_bindings.json').read_text()))
rows=[]; patches=[]
for rel in files:
 base=R/rel; dst=O/'base'/rel; dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(base.read_bytes())
 proc=subprocess.run(['git','merge-file','-p',str(P/'proposed'/rel),str(P/'base'/rel),str(dst)],capture_output=True)
 out=O/'proposed'/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(proc.stdout)
 rows.append({'path':rel,'merge_exit':proc.returncode,'merge_stderr':proc.stderr.decode(),'live_base_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'proposed_sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
 patches+=difflib.unified_diff(dst.read_text().splitlines(True),out.read_text().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)
(O/'initial_merge_inventory.json').write_text(json.dumps(rows,indent=2)+'\n');(O/'minimal.patch').write_text(''.join(patches))
print(json.dumps(rows,indent=2))
