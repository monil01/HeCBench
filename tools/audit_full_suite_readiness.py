#!/usr/bin/env python3
"""Inventory source-backed readiness signals without claiming a native build/run."""
import datetime,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT/'.porting-state/portable_julia';batch=json.loads((S/'batch_537.json').read_text());rows=[]
for b in batch['new_benchmarks']:
 p=ROOT/'src'/(b+'-cuda');source=[x for x in p.rglob('*') if x.is_file() and x.suffix in ['.cu','.cuh','.h','.hpp','.cpp','.cc','.c']];controls=[x for x in [p/'Makefile',p/'CMakeLists.txt',p/'README.md',p/'README'] if x.exists()];texts={str(x.relative_to(ROOT)):x.read_text(errors='replace') for x in source+controls};signals={}
 for name,regex in {'mpi':r'\bMPI_|\bmpi\.h\b|-lmpi\b','nccl':r'\bnccl[A-Z]|nccl\.h|-lnccl\b','vendor_blas':r'\bcublas|\bcusparse','rng':r'\bcurand','pytorch':r'\btorch\b|ATen|TensorAccessor','file_input':r'\bfopen\s*\(|\bifstream\b|UNZIPFILE','inline_ptx':r'asm\s*(?:volatile)?\s*\('}.items():
  evidence=[]
  for path,text in texts.items():
   for n,line in enumerate(text.splitlines(),1):
    if re.search(regex,line):evidence.append({'path':path,'line':n,'text':line[:250]})
  signals[name]={'observed':bool(evidence),'evidence':evidence[:30],'truncated':len(evidence)>30}
 rows.append({'benchmark':b,'owner':batch['ownership'].get(b),'status':'source_inventory_only_execution_unassessed','julia_directory_exists':(ROOT/'src'/(b+'-julia')).is_dir(),'code_files':len(source),'code_lines':sum(len(t.splitlines()) for path,t in texts.items() if path not in {str(x.relative_to(ROOT)) for x in controls}),'dvc_manifests':[str(x.relative_to(ROOT)) for x in p.rglob('*.dvc')],'build_controls':[{'path':str(x.relative_to(ROOT)),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in controls],'signals':signals,'native_build_tested':False,'native_execution_tested':False,'missing_assets_confirmed':None,'candidate_attempts':None,'token_actuals':None,'scope':'Static matches are review leads, never terminal barriers, unsupported features, missing-input proof or executed-work measurements.'})
out=ROOT/'2026-PASC-Hecbench-Julia-paper/research/expansion_537/readiness_inventory.json';out.write_text(json.dumps({'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'population':537,'frozen_assessed':447,'new_names':len(rows),'signal_counts':{k:sum(r['signals'][k]['observed'] for r in rows) for k in rows[0]['signals']},'rows':rows},indent=2)+'\n');print(json.dumps({'new_names':len(rows),'source_only':True,'signal_counts':json.loads(out.read_text())['signal_counts']}))
