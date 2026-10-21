"""CPU-only final binding refresh; never loads/executes authored runner/helpers."""
from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta=lambda p:{'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
d=json.loads((P/'proposal.json').read_text());extra=json.loads(Path('/tmp/roofline-completion-20261021/allreduce_capture_adapter/prospective_parser_calibration_bindings.json').read_text())
for path,digest in extra.items():
 if sha(path)!=digest:raise ValueError('prospective binding changed '+path)
d['all_readonly_bindings'].update(extra)
calibration=Path('/home/imo/HeCBench_julia/ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json');d['calibration_review']=meta(calibration)
d['prospective_calibration_bindings']=extra;d['snapshot_review_paths']=[str(P/'source_equivalence.json'),str(P/'documentation_evidence.json'),str(calibration)]
d['device_attestation_contract']={'label':'ALLREDUCE_DEVICE_ATTEST','phases':['before_first_warmup_add','after_all5_full_checks'],'exact_count':4,'per_rank_count':2,'fields':['rank','logical_device','pid','pci','uuid','phase'],'logical_device_equals_rank':True,'rank0_physical_device':5,'rank1_physical_device':6,'CUDA_runtime_UUID_and_PCI_checked_against_actual_snapshot':True,'full_program_activity_mapping':'every observed registered rank0→GPU5,rank1→GPU6; profilerGPU6only; unexpected startup context fails, no grace'}
e=json.loads((P/'source_equivalence.json').read_text())
for model,record in e.items():record['host_marker_header']=meta(P/'profile_window.h');record['host_only_attestation_addition']='Before first warmup add and after all5fullchecks, query current CUDA device/PCI/UUID and assert exact rank/device mapping; unchanged add kernel and MPI algorithm bytes. No build/runtime observation.'
(P/'source_equivalence.json').write_text(json.dumps(e,indent=2)+'\n')
for model,rec in d['models'].items():
 rec['execution_bindings'].update(extra)
 rec['host_marker_header']=meta(P/'profile_window.h')
 for path in list(rec['execution_bindings']):
  if str(P)in path and Path(path).is_file():rec['execution_bindings'][path]=sha(path)
for path in list(d['all_readonly_bindings']):
 if str(P)in path and Path(path).is_file():d['all_readonly_bindings'][path]=sha(path)
d['runner']=meta(P/'run_proposal.py');d['source_equivalence_proof']=meta(P/'source_equivalence.json')
d['integration_proposal']={'directory':'/tmp/roofline-completion-20261021/allreduce_capture_adapter','status':'PRIVATE_ADAPTER_REQUIRES_INDEPENDENT_ROOT_REVIEW_AND_EXPLICIT_INGESTION_ADOPTION','source_authority':'actual copied runner, not generic collector','capture_prevalidation_snapshots':['collector_source.py','parser_source.py','calibration_review.json','source_snapshots.json','snapshot_bindings.json'],'readonly_bindings_before_after_explicit':True,'no_shared_ledger_or_summarizer_change':True}
# Validate readonly file metadata only, not implementation behavior.
for path,digest in d['all_readonly_bindings'].items():
 if sha(path)!=digest:raise ValueError('final binding mismatch '+path)
(P/'proposal.json').write_text(json.dumps(d,indent=2)+'\n')
r=json.loads((P/'v1_frozen/READY_FOR_ROOT_REVIEW.json').read_text());r['revision']=2;r['created_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();r['proposal']=meta(P/'proposal.json');r['runner']=meta(P/'run_proposal.py');r['source_equivalence']=meta(P/'source_equivalence.json');r['source_hash_bindings_checked_count']=len(d['all_readonly_bindings']);r['v1_preserved']=meta(P/'v1_frozen/FREEZE_MANIFEST.json')
for item in r['root_review_entry_points']:
 item.update(meta(Path(item['path'])))
r['root_review_entry_points'].append(meta(P/'frozen_v100_roofline_classify.py'))
r['revision_changes']={'metrics':'Exact root27 standard metrics +3PCI+2CC; exact frozen parser, all required numeric finite nonnegative/units, durationpositive, rawsm70/PCI identity. Automatic strings retain parserNone; CSV bytes preserved.','rank_identity':'Registered rank0GPU5/rank1GPU6 exclusively, profilerGPU6only plus4runtime CUDA UUID/PCI attests. No Julia startup context grace.','queue':'Configurable1..86400s lock/idle waits outside permodel1800s observer/build budget. No foreign signals.','NCU_ELF':'Direct exact wrapper x86_64 target to preserve registeredPID; wrapper and target both SHA-bound, no wildcard. Requires root review as host launcher adaptation.','integration':'Private adapter proposal, prospective parser/collectorreference/calibration supportingfiles bound. Pre-capture source snapshots plus exact validatedreadonlybefore/after maps. Existing shared ingestion unchanged.'}
r['counter_contract']=d['counter_contract'];r['device_attestation_contract']=d['device_attestation_contract'];r['integration_proposal']=d['integration_proposal'];r['queue_policy']=d['queue_policy'];r['ncu_direct_ELF_adaptation']=d['ncu_direct_ELF_adaptation']
r['future_execution_command_template']+=['--queue-wait-seconds','86400'];r['pending_prerequisites'].append('Root independent source review/adoption of private adapter and method-specific ontology/audit integration; adapter/runtime not executed or validated. No measurement acceptance until actual capture passes all gates.')
(P/'READY_FOR_ROOT_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n')
(P/'REVISION_V2.json').write_text(json.dumps({'revision':2,'root_reviewed_v1_preserved':meta(P/'v1_frozen/FREEZE_MANIFEST.json'),'proposal':meta(P/'proposal.json'),'runner':meta(P/'run_proposal.py'),'review':meta(P/'READY_FOR_ROOT_REVIEW.json'),'changes':r['revision_changes'],'source_only_review_no_tests':True,'runner_executions':0,'GPU_queries':0,'GPU_runs':0,'builds':0,'tests':0,'GPU5_authorized':False},indent=2)+'\n')
print(json.dumps({'proposal':r['proposal'],'runner':r['runner'],'review':meta(P/'READY_FOR_ROOT_REVIEW.json'),'bindings':len(d['all_readonly_bindings'])}))
