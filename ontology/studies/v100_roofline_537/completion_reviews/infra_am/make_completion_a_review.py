from pathlib import Path
import json,hashlib
S=Path('/home/imo/HeCBench_julia');D=Path(__file__).parent;old=json.loads((D/'source_overlay_proposals.json').read_text());by={x['case']:x for x in old['cases']};base=D/'current_base_v7';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();slots=[];sources={}
case_models=[('ccs_native_tmpfile','ccs',['cuda']),('cm_jacc_trace','cm',['jacc']),('miniDGS_persist','miniDGS',['cuda','cuda_jl','ka','jacc']),('mf_sgd_native_precreate','mf-sgd',['cuda']),('bn_native_failure_capture','bn',['cuda']),('clock_native_outputs','clock',['cuda']),('gc_outputs','gc',['cuda','cuda_jl','ka','jacc']),('memcpy_native_output','memcpy',['cuda'])]
for case,b,models in case_models:
 for m in models:
  p=base/b/m/'plan.json';plan=json.loads(p.read_text());row={'benchmark':b,'model':m,'case':case,'base_plan':str(p),'base_plan_sha256':sha(p),'clear_gates':plan['blocking_gates'],'reason':'Exact reviewed host-only capture-root/fresh-output/readonly-input lookup from infra_am source review. No source device math/geometry/RNG/precision/iterations/tolerances or scientific checks change.'}
  if case!='mf_sgd_native_precreate':
   f=by[case]['files'][0]
   if case=='gc_outputs':f=by[case]['files'][0 if m=='cuda'else 1]
   row['source']=f['path'];row['replacements']=f['replacements']
  slots.append(row)
b='columnarSolver';m='cuda';p=D/'current_base_v7_columnar'/b/m/'plan.json';slots.append({'benchmark':b,'model':m,'case':'columnar_native_compiler','base_plan':str(p),'base_plan_sha256':sha(p),'clear_gates':['source_bound_absolute_path_literals_require_write_isolation_review'],'reason':'Source run_native.py readonly P/gold preserved; exclusive fresh native_captures root and fresh executable. Same accepted CUDA12.0 O3 C++17 sm70 DATA_PATH command flags from earlier approved host overlay, replacing nonexistent /usr/local/cuda/bin/nvcc with exact installed /usr/bin/nvcc wrapper ->CUDA12.0.140.','reviewed_compiler_command':['/usr/bin/nvcc','-O3','-std=c++17','-arch=sm_70','-DDATA_PATH="'+str(S/'src/columnarSolver-cuda/data')+'"',str(S/'src/columnarSolver-julia/study_native_storage_v2.cu'),'-o','${SCRATCH}/study_native_storage_v2']})
for b in {r['benchmark']for r in slots}:
 for directory in [S/'src'/(b+'-julia'),S/'src'/(b+'-cuda')]:
  for p in directory.rglob('*'):
   if p.is_file()and p.suffix in ['.jl','.py','.cu','.cpp','.c','.cuh','.h','.hpp','.inc','.toml']:sources[str(p.resolve())]=sha(p)
for p in [S/'src/clock-julia/fixture.bin',S/'src/gc-julia/fixture_csr.bin',S/'src/miniDGS-julia/libstudy_mpi.so',S/'src/memcpy-julia/pattern_u32.bin',Path('/usr/bin/nvcc'),Path('/usr/lib/nvidia-cuda-toolkit/bin/nvcc'),S/'ontology/studies/v100_roofline_537/approved_isolation_reviews_v7.json',D/'source_overlay_proposals.json',D/'non_source_retry_proposals.json']:
 sources[str(p.resolve())]=sha(p)
(D/'completion_a_review.json').write_text(json.dumps({'scope':'CPU-only profile-only output isolation; exact current v7 base plans and reviewed canonical readonly source/data. No target runs or build performed.','slots':slots,'source_bindings':sources,'excluded_pending_review':{'egs':'Runtime compiler exit255 cause not established; no automatic private childbuildenv change.','bitcracker':'Provider issue separately reviewed by rootretrydriver021, not staged here.','assert':'Existing study/assert_review already supplies separate methods.'}},indent=2)+'\n');print(len(slots),'slots',len(sources),'bindings',sha(D/'completion_a_review.json'))
