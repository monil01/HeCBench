# CPU ELF/header/file metadata only. Never imports NumPy or executes any target/library.
from pathlib import Path
import json,hashlib,subprocess,re
O=Path(__file__).resolve().parent;BASE=Path('/tmp/pasc-julia-plot-tools');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();files={};elfs=[]
for directory in [BASE/'numpy',BASE/'numpy.libs',BASE/'numpy-2.5.4.dist-info']:
 for p in sorted(directory.rglob('*')):
  if p.is_file():files[str(p.resolve())]=H(p)
for p in list(files):
 if '.so'in Path(p).name:elfs.append(Path(p))
elfs.append(Path('/usr/bin/python3').resolve());reports=[];visited=set();unresolved=[]
while elfs:
 p=elfs.pop()
 if str(p)in visited:continue
 visited.add(str(p));files[str(p)]=H(p);cmd=['/usr/bin/readelf','-d',str(p)];r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,(str(p),r.stderr);needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',r.stdout);resolved=[]
 for n in needed:
  candidates={q.resolve()for q in [p.parent/n,BASE/'numpy.libs'/n,Path('/usr/lib/x86_64-linux-gnu')/n,Path('/lib/x86_64-linux-gnu')/n,Path('/lib64')/n]if q.is_file()}
  if len(candidates)!=1:unresolved.append({'ELF':str(p),'needed':n,'candidates':list(map(str,candidates))});continue
  q=candidates.pop();files[str(q)]=H(q);elfs.append(q);resolved.append({'name':n,'path':str(q),'sha256':H(q)})
 reports.append({'ELF':str(p),'ELF_sha256':H(p),'read_command':cmd,'dynamic_metadata_stdout':r.stdout,'stderr':r.stderr,'needed_resolved':resolved})
assert not unresolved,unresolved
metadata=BASE/'numpy-2.5.4.dist-info/METADATA';assert 'Version: 2.5.4\n'in metadata.read_text()
(O/'source/numpy_bindings.json').write_text(json.dumps({'expected_numpy_version':'2.5.4','package_roots':[str(BASE/'numpy'),str(BASE/'numpy.libs'),str(BASE/'numpy-2.5.4.dist-info')],'files':files,'scope':'Exact existingpackage/distribution/extension/shared-library dependency/CPUinterpreter bytes from filemetadata; actual future runtimeidentity separately recorded, not inferred for original primaryoracle.','metadata_only_no_numpy_import_or_install':True},indent=2)+'\n');(O/'numpy_elf_dependency_review.json').write_text(json.dumps({'unresolved':unresolved,'file_count':len(files),'ELF_metadata_records':reports,'no_dependency_or_package_execution':True,'no_OS_package_changes':True},indent=2)+'\n');print('pinned_files',len(files),'ELFs',len(reports))
