from pathlib import Path
import hashlib,json,difflib
D=Path(__file__).resolve().parent;R=Path('/tmp/hecbench-roofline-v100-20261020/batch040-miniDGS/miniDGS');J=Path('/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia')
def sha(b):return hashlib.sha256(b).hexdigest()
old=b"rankcmd=['julia','--startup-file=no'";new=b"rankcmd=['"+str(J).encode()+b"','--startup-file=no'"
parents={};reference=None
for m in ['cuda_jl','ka','jacc']:
 p=R/m/'plan.json';plan=json.loads(p.read_bytes());s=R/m/'scratch/profile_source/study_driver.py';raw=s.read_bytes()
 if reference is None:reference=raw
 if raw!=reference or raw.count(old)!=1 or new in raw:raise RuntimeError('Exact identical one-token parent required')
 parents[m]={'plan_path':str(p),'plan_sha256':sha(p.read_bytes()),'source_path':str(s),'source_sha256':sha(raw),'plan_original_config_sha256':plan['completion_overlay_provenance']['original_config_sha256'],'plan_original_result_sha256':plan['completion_overlay_provenance']['original_result_sha256']}
patched=reference.replace(old,new,1)
if patched.count(new)!=1 or patched.replace(new,old,1)!=reference:raise RuntimeError('Inverse byte identity failed')
(D/'study_driver.parent.py').write_bytes(reference);(D/'study_driver.absolute_julia.py').write_bytes(patched)
(D/'path_only.patch').write_text(''.join(difflib.unified_diff(reference.decode().splitlines(True),patched.decode().splitlines(True),fromfile='040/study_driver.py',tofile='prospective/study_driver.py')))
report={'status':'SOURCE_ONLY_COPY_NOT_EXECUTED_NO_PLANS_AUTHORED','parents':parents,'old_source_sha256':sha(reference),'new_source_sha256':sha(patched),'exact_inverse_byte_identity':True,'replacement_occurrences':1,'replaced_bytes':old.decode(),'replacement_bytes':new.decode(),'julia_executable':{'path':str(J),'sha256':sha(J.read_bytes())},'preserved_by_inverse_byte_proof':['MPI2 command and rank semantics','GPU6 physical assignment','canonical readonly P fixture and source root','model projects and target main.jl','all5 trials/rank correctness/oracle/input and output captures','capture paths derived from STUDY_OUTPUT_DIR','all numerical code, timers, precision, geometry, tolerances and stages'],'native_driver_not_to_patch':True,'prospective_plan_binding_updates_required':['Fresh target argv copied-driver path; original accepted config/result references unchanged','Fresh execution_bindings KEY for copied driver -> new SHA; retain original canonical driver SHA as readonly provenance; no old historical source SHA relabel','Bind absolute Julia ELF fd670aabc838e93f178cbcf7304c5e9c50aadcce3735e86d5e3218b1bb04e602, rehash before/after future capture','Retain original overlay chain; append new original/copy/diff SHA plus full new source text and explicit path-only reason','Rebase fresh scratch/output/TMPDIR/build-guard destinations and selector/review paths with exact-source root review; all other bindings unchanged','Retain miniDGS process-filter selector and conditional replay review; update source-copy references in supplemental context and any nested execution map, without editing archived040','Actual003 all9 success + root v5 immutable method review and new tool bindings prerequisite; no receipt fabricated here','Wait predecessor040 and recheck selected qualification; skip-qualified, finite source-only authoring does not launch collector'],'gpu_runs':0,'builds':0,'imports_of_target':0,'tests':0,'shared_mutations':0}
for m,rec in parents.items():
 if sha(Path(rec['source_path']).read_bytes())!=rec['source_sha256']or sha(Path(rec['plan_path']).read_bytes())!=rec['plan_sha256']:raise RuntimeError('Parent changed during review')
(D/'review.json').write_text(json.dumps(report,indent=2)+'\n')
