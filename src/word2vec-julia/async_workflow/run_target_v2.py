#!/usr/bin/env python3
"""Central-verifier model command; all raw files persist under a short owned path."""
import os,pathlib,json,subprocess,sys,hashlib
import numpy as np
from check_target_properties import check
P=pathlib.Path(__file__).resolve().parent
base=pathlib.Path('/tmp/w2vfull')/(sys.argv[1]+'_'+str(os.getpid()));base.mkdir()
protocol=json.loads((P/'protocol.json').read_text());proof={'protocol_sha256':hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest(),'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [P/'study_bridge_v2.cu',P/'observed_host.inc',P/'target_gpu.inc',P/'check_target_properties.py',P/'run_target_v2.py',pathlib.Path(sys.argv[4]),P/'workflow_driver.jl']},'storage_root':str(base),'raw_model_identity_claim':False,'before_execution':True};(base/'before.json').write_text(json.dumps(proof,indent=2)+'\n')
r=subprocess.run([sys.argv[2],'--startup-file=no','--project='+sys.argv[3],sys.argv[4],'/tmp/w2vfull/libworkflow_v2.so',str(base)]);print('RAW_WORKFLOW_ROOT '+str(base),flush=True)
if r.returncode:raise SystemExit(r.returncode)
allgood=True
for t in range(6):
 folder=base/f'trial{t}';r=check(folder);allgood=allgood and r['pass'];timing=json.loads((folder/'timing.json').read_text())
 # Lossless static inputs including RLE of full monotonic table; certificate only.
 table=np.memmap('/tmp/w2vfull/table.i32',dtype='<i4',mode='r');hist=np.bincount(table,minlength=1025)
 blocks=[np.fromfile(folder/'initial_syn0.f32','<f4'),np.fromfile(folder/'initial_syn1.f32','<f4'),np.fromfile(folder/'initial_rng.u32','<u4'),np.fromfile(folder/'initial_sen.i32','<i4'),np.fromfile(folder/'exptable.f32','<f4'),hist,np.array([int(x.split('\t')[1])for x in (folder/'vocab.tsv').read_text().splitlines()])]
 cert=np.concatenate(blocks).astype('<f8');path=pathlib.Path(os.environ.get('STUDY_OUTPUT_DIR','.'))/f'output_trial{t}.bin';cert.tofile(path)
 row={'trial':t,'time_seconds':timing['time_seconds'],'correct':r['pass'],'checked_values':int(len(cert)),'max_abs_error':None,'max_rel_error':None,'timing_region':protocol['timing_region'],'jit_excluded':True,'internal_repeats':1,'output_artifact':{'path':str(path),'dtype':'float64','count':int(len(cert)),'byte_order':'little'},'raw_model_identity_claim':False,'numerical_agreement_claim':False,'artifact_role':'lossless_static_input_only_property_audit_separate_not_rawmodel_output','raw_full_workflow_root':str(folder),'independent_property_audit_sha256':hashlib.sha256((folder/'properties.json').read_bytes()).hexdigest(),'raw_output_complete':True,'loss_improvement':r['checks']['fixed_expected_loss_improvement']['improvement'],'callback_count':r['checks']['trace_complete']['callbacks']}
 if t:print('STUDY_TRIAL '+json.dumps(row),flush=True)
 else:print('WARMUP_PROPERTY '+json.dumps({'correct':r['pass'],'failures':r['failures']}),flush=True)
print('ALL_FULL_WORKFLOW_PROPERTIES '+str(allgood),flush=True)
raise SystemExit(0 if allgood else 1)
