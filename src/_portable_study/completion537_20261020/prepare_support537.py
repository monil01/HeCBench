from pathlib import Path
import json,hashlib
R=Path('/home/imo/HeCBench_julia');P=R/'src/_portable_study/completion537_20261020';names=set(json.loads((P/'source_commit_receipts.json').read_text())['source_commits'])|{'tpacf'};files=set();essential=[];skipped=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
text_ext={'.jl','.cu','.cuh','.h','.hpp','.cpp','.c','.inc','.toml','.md','.json','.py','.sh','.txt','.tsv','.csv','.diff','.patch','.ptx','.cmake','.dvc'}
def add(f,root,explicit=False):
 if not f.is_file()or not f.resolve().is_relative_to(root.resolve()):return
 if any(x in f.parts for x in ['__pycache__','.git']):return
 if f.stat().st_size>50*1024*1024:skipped.append({'path':str(f.relative_to(R)),'reason':'Large local evidence excluded fromsourceGit; retained on disk'});return
 if f.suffix in {'.so','.o','.a','.pyc','.bin','.avi','.tar','.gz','.bz2','.zip','.flo'}:return
 if not explicit and f.suffix not in text_ext and f.name not in ['README','Makefile','LICENSE','COPYING','COPYRIGHT','NOTICE','CMakeLists.txt']:return
 with f.open('rb')as s:head=s.read(1024)
 if head.startswith(b'\x7fELF')or b'\0'in head:return
 files.add(str(f.relative_to(R)))
def inventory_paths(value,root,explicit=True):
 if isinstance(value,dict):
  for k,v in value.items():
   if k in ['exclude','excluded','regenerable_binary_files','regenerable_binary_inputs']:continue
   inventory_paths(v,root,explicit)
 elif isinstance(value,list):
  for v in value:inventory_paths(v,root,explicit)
 elif isinstance(value,str) and len(value)<1000 and '\n'not in value:
  f=R/value
  if not f.exists():f=root/value
  if f.exists():add(f,root,explicit)
for n in sorted(names):
 root=R/f'src/{n}-julia';inventory=root/'ready_support_files.json'
 if inventory.exists():inventory_paths(json.loads(inventory.read_text()),root)
 else:
  for f in root.rglob('*'):
   if any(x in f.relative_to(root).parts for x in ['dataset_readiness','original_assets','raw','build','environments','data','cas','CAS']):continue
   add(f,root)
 # Current root metadata and preserved accounting deliveries are source records, never payload arrays.
 for f in root.glob('*.json'):add(f,root)
 for f in (root/'reference_history/accounting_original_delivery_20261020').rglob('*.json'):add(f,root)
 manifest=root/'essential_assets_for_commit.json'
 if manifest.exists():
  d=json.loads(manifest.read_text())
  def required(v):
   if isinstance(v,dict):
    if isinstance(v.get('path'),str) and v.get('sha256'):
     f=R/v['path'];f=f if f.exists()else root/v['path'];assert f.is_file() and f.resolve().is_relative_to(root.resolve());assert sha(f)==v['sha256'];assert f.stat().st_size==v.get('bytes',f.stat().st_size);assert f.stat().st_size<50*1024*1024;files.add(str(f.relative_to(R)));essential.append({'path':str(f.relative_to(R)),'bytes':f.stat().st_size,'sha256':sha(f)})
    else:
     for k,x in v.items():
      if k not in ['exclude','additional_small_expected_input_frames']:required(x)
   elif isinstance(v,list):
    for x in v:required(x)
  required(d)
 if n=='lsqt':
  for f in sorted(root.glob('*_fixture.bin'))+sorted(root.glob('independent_math_oracle.bin*')):
   assert f.stat().st_size<50*1024*1024;files.add(str(f.relative_to(R)));essential.append({'path':str(f.relative_to(R)),'bytes':f.stat().st_size,'sha256':sha(f),'source':'Explicitworkeressentialassets handoff; CPUregeneration/provenance source-bound'})
for f in (R/'src/tsne-julia/guarded_I1').rglob('*'):
 if f.suffix in text_ext:add(f,R/'src/tsne-julia')
for f in (R/'src/_portable_study/stencil_ports_537_audit_20261020').glob('*'):add(f,R/'src/_portable_study')
for f in P.glob('*'):add(f,R/'src/_portable_study')
for n in ['AGENT_INSTRUCTIONS_CODEX_Julia.md','JULIA_CORRECTNESS_CHECK.md','paper_writing.md','tools/verify_portable_julia.py','tools/commit_verified_portable_variant.py','tools/export_full_suite_expansion.py','tools/study_evidence.py']:files.add(n)
report={'benchmarks':sorted(names),'source_support_paths':sorted(files),'essential_asset_bytes':sum(e['bytes']for e in essential),'essential_assets':essential,'skipped_large_source_files':skipped,'scope':'Explicitreadyinventories or bounded source-only fallback; requiredsmallinputs verified. No rawGPUcaptures/research/compiledobjects/caches/externalvideos. Separateimplementationcommits precede thissupportjoin.'};(P/'support_staging_review.json').write_text(json.dumps(report,indent=2)+'\n');print(len(names),'benchmarks;',len(files),'files;',report['essential_asset_bytes'],'essential bytes; skipped',len(skipped))
