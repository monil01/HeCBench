from pathlib import Path
import json,hashlib,fcntl,struct,shutil
P=Path(__file__).parent;S=Path('/tmp/hecbench-roofline-v100-20261020/batch071-sparkler-privateMPI/sparkler/cuda');W=Path('/home/imo/HeCBench_julia');H=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/2025/comm_libs/12.9/hpcx/hpcx-2.22.1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
bindings={}
for n in ['execution.log','counters.csv','profile_provenance.json','plan.json','original_plan.json','profiling_auxiliary_manifest.json','write_guard_source.py','collector_source.py','profiler_method_review.json','profiler_method_post_profile_check.json','counter_quiescence_snapshot.json','build.log']:
 q=S/n;dest=P/'actual071'/n;dest.parent.mkdir(exist_ok=True);shutil.copyfile(q,dest);bindings[str(q)]={'sha256':sha(q),'bytes':q.stat().st_size}
p=json.loads((S/'plan.json').read_bytes());prov=json.loads((S/'profile_provenance.json').read_bytes());log=(S/'execution.log').read_text();manifest={x['path']:x for x in json.loads((S/'profiling_auxiliary_manifest.json').read_bytes())}
checks={'actual_profile_rc139':prov['profile']['returncode']==139,'collector_stop_reason_null':prov['profile']['stop_reason'] is None,'actual_build_zero':prov['build']['returncode']==0,'profile_plan_sha_join':sha(S/'plan.json')==prov['plan_sha256'],'execution_log_sha_join':sha(S/'execution.log')==prov['profile']['log_sha256'],'manifest_sha_join':sha(S/'profiling_auxiliary_manifest.json')==prov['auxiliary_manifest_sha256'],'no_complete_trials':not any(x.startswith('STUDY_TRIAL ') for x in log.splitlines()),'all4_ucx_shm_open_denials':log.count('shm_open(file_name=/ucx_shm_posix_')==4,'all4_ucp_worker_failure':log.count('TL_UCP ERROR failed to create ucp worker')==4,'all4_ucc_context_failure':log.count('UCC  ERROR failed to create tl context for ucp')==4,'all4_SIGSEGV_nil':log.count('Signal: Segmentation fault (11)')==4 and log.count('Failing at address: (nil)')==4,'no_MPI_vendor_markers':'SPARKLER_MPI ' not in log and 'SPARKLER_VENDOR ' not in log,'counter_no_ID_header':'"ID","Process ID"' not in (S/'counters.csv').read_text(),'four_private_MPI_backings_manifested':sum(x.startswith('ncu_tmp/sparkler_mpi/backing/open_mpi_shmem_mmap.') for x in manifest)==4}
patches=p['sparkler_private_MPI_recovery']['patches'];source_rows=[]
for x in patches:
 data=x['after_text'].encode();h=hashlib.sha256(data).hexdigest();relative='profile_source/'+Path(x['copy']).name
 assert h==x['copy_sha256']==manifest[relative]['sha256']==p['execution_bindings'][x['copy']]
 dest=P/Path(x['copy']).name;dest.write_bytes(data);source_rows.append({'path_role':'exact071retained_plan_source_text_not_historical_scratch_recreation','original_scratch_path':x['copy'],'frozen_path':str(dest),'sha256':h,'manifest_full_source_SHA_equal':True})
launcher=(P/'study_launch.py').read_text();needle="env.update(OMPI_MCA_orte_tmpdir_base=str(sessions),OMPI_MCA_shmem_mmap_relocate_backing_file='1',OMPI_MCA_shmem_mmap_backing_file_base_dir=str(backing),OMPI_MCA_btl_vader_backing_directory=str(vader))\n"
addition="ucx_posix=mpi_base/'ucx_posix';ucx_posix.mkdir(mode=0o700)\nenv['UCX_POSIX_DIR']=str(ucx_posix)\n"
assert launcher.count(needle)==1;future=launcher.replace(needle,needle+addition);assert future.replace(addition,'')==launcher
(P/'future_launcher_path_only_hypothesis.py').write_text(future)
dump(P/'future_launcher_inverse.json',{'status':'mechanism_supported_actual_recovery_untested_not_launch_authority','before_sha256':sha(P/'study_launch.py'),'after_sha256':sha(P/'future_launcher_path_only_hypothesis.py'),'only_added_text':addition,'exact_byte_inverse_equal':future.replace(addition,'').encode()==(P/'study_launch.py').read_bytes(),'UCX_UCC_collective_transport_flags_unchanged':True,'USE_PROC_LINK_default_y_unchanged':True})
with Path('/tmp/hecbench-roofline-metadata.lock').open('r') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);ledger=W/'ontology/studies/v100_roofline_537/benchmarks/sparkler.json';r=ledger.read_bytes();contract=json.loads(r)['accepted_named_contract'];bindings[str(ledger)]={'sha256':hashlib.sha256(r).hexdigest(),'bytes':len(r)}
 checks['current_CID_join']=contract['configuration_id']==p['configuration_id'];checks['current_result_join']=str(W/contract['result'])==p['reference_result'] and contract['result_sha256']==p['reference_result_sha256'];checks['current_config_join']=contract['raw_config_sha256']==p['raw_config_sha256']
for q,h in [(Path(p['reference_result']),p['reference_result_sha256']),(W/contract['result'].replace('result.json','config.json'),p['raw_config_sha256'])]:
 checks['current_file_'+q.name+'_SHA']=sha(q)==h;bindings[str(q)]={'sha256':sha(q),'bytes':q.stat().st_size}
# Canonical source hashes remain explicit; deleted scratch paths are covered by
# retained full plan text + successful full auxiliary manifest, never fabricated.
canonical=[]
for name,h in p['execution_bindings'].items():
 q=Path(name)
 if name.startswith(str(W/'src')) and q.is_file():canonical.append({'path':name,'expected_sha256':h,'current_sha256':sha(q),'matches':sha(q)==h})
checks['current_canonical_explicit_source_bindings']=all(x['matches'] for x in canonical)
lib=H/'ucx/lib/libuct.so.0.0.0';b=lib.read_bytes()
def string(addr):
 if not addr:return None
 offset=addr if 0x2b000<=addr<0x317b9 else addr-0x1000
 return b[offset:b.index(b'\0',offset)].decode()
entries=[]
for i in range(5):
 addr=0x3cb40+88*i;values=struct.unpack_from('<11Q',b,addr-0x1000);entries.append({'record_vaddr':hex(addr),'name':string(values[0]),'default':string(values[1]),'documentation':string(values[2]),'configuration_offset':values[3],'raw_record_hex':b[addr-0x1000:addr-0x1000+88].hex()})
assert entries[1]['name']=='DIR' and entries[1]['default']=='/dev/shm';assert entries[3]['name']=='USE_PROC_LINK' and entries[3]['default']=='y'
installed=[lib,H/'ucx/include/ucs/config/parser.h',H/'ucx/include/ucp/api/ucp.h',H/'ompi/lib/openmpi/mca_coll_ucc.so',H/'ucc/lib/libucc.so.1.0.0',H/'ucx/lib/libucp.so.0.0.0',H/'ucx/lib/libucs.so.0.0.0']
installed_rows=[{'path':str(q),'resolved':str(q.resolve()),'sha256':sha(q),'bytes':q.stat().st_size,'scope':'Current installed bytes for prospective mechanism proof; not an actual071 loaded-DSO identity receipt'} for q in installed]
dump(P/'installed_UCX_mechanism.json',{'scope':'Static source/header/ELF bytes only; no library loading or execution','files':installed_rows,'ELF_layout':{'rodata_vaddr_equals_file_offset':True,'data_file_offset_equals_vaddr_minus':'0x1000','uct_posix_md_config_table_vaddr':'0x3cb40','entry_size_bytes':88,'component_vaddr':'0x3ce60','component_prefix_pointer_vaddr':'0x2e1af','component_table_pointer_vaddr':'0x3cb40'},'component_prefix':string(0x2e1af),'config_table':entries,'header_env_prefix':'UCX_','proposed_exact_variable':'UCX_POSIX_DIR','default_PROC_LINK_not_modified':True,'historical_loaded_library_identity_proven':False})
dump(P/'evidence_audit.json',{'status':'saved_failure_source_review_not_runtime_recovery','checks':checks,'all_checks_pass':all(checks.values()),'actual_files':bindings,'source_fulltext_joins':source_rows,'canonical_source_bindings':canonical,'source_primary_contract':contract,'classification':{'observed':'Four ranks fail POSIX shm_open with Permission denied, UCP/UCC context errors, then four null-address SIGSEGV; MPI rc139; no scientific trials/rawcounter header','proximate_diagnostic':'MPI collective startup UCC/UCX POSIX shared-memory path cannot create backing under active guard policy; prior MPI mmap backing redirection itself produced four valid full manifests','inference':'/dev/shm create denial is consistent with the exact Landlock allowlist, which excludes /dev/shm. No syscall trace isolates Landlock from all other permission causes.','unique_segmentation_fault_root_cause':'Unknown: saved stack shows only libc signal frame, not the failing UCC/UCX instruction or full call chain','future_hypothesis':'Exact UCX_POSIX_DIR fresh private directory override; same UCC/UCX implementation and MPI graph; actual recovery must be tested separately after root review','unsupported_profiler_method_proven':False,'current_failure_admitted':False}})
assert all(checks.values()),checks
print(json.dumps({'checks':len(checks),'all_pass':True,'static_UCX_sha256':sha(lib)}))
