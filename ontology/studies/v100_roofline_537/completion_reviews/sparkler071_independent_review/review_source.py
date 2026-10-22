from pathlib import Path
import json,gzip,hashlib,copy,fcntl,datetime,os
O=Path(__file__).resolve().parent;P=Path('/tmp/roofline-completion-20261021/sparkler071_recovery');W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';SRC=W/'src/sparkler-julia';UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d';checks=[];models=[];sources=[];aliases=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def ck(x,name):
 if not x:raise RuntimeError(name)
 checks.append(name)
def B(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def relocate(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return[relocate(v,a,b)for v in x]
 if isinstance(x,dict):return{k:relocate(v,a,b)for k,v in x.items()}
 return x
D=P/'DELIVERY.json';ck(sha(D)=='91c56f9a007250bd3922c156ac2d6acde1902b4a58addc622bd27dfb9ebc25d1','exactDELIVERY')
for e in load(D)['files']:ck(Path(e['path']).stat().st_size==e['bytes']and sha(e['path'])==e['sha256'],'deliveryfile:'+e['path'])
mpi=load(P/'installed_MPI_parameter_evidence.json')
for e in mpi['evidence']:
 raw=Path(e['path']).read_bytes();ck(sha(e['path'])==e['sha256'],'MPIinstalledSHA:'+e['path'])
 for q in e.get('evidence',[]):ck(raw[q['start_byte']:q['end_byte']].decode()==q['NUL_delimited_installed_component_text']and q['needle']in q['NUL_delimited_installed_component_text'],'MPIexactbyteexcerpt:'+q['needle'])
 for q in e.get('line_evidence',[]):ck(raw.decode().splitlines()[q['line']-1]==q['text'],'MPIexactmanline:'+str(q['line']))
ck('Positive = same as the negative option, but will fail if the relocation fails.'in mpi['evidence'][0]['evidence'][0]['NUL_delimited_installed_component_text'],'MPIpositive1fails_not_negativefallback')
marker="cmd=[str(H/'ompi/bin/mpirun'),'--bind-to','none','-n','4']+sys.argv[1:]"
insert="mpi_base=Path(env['TMPDIR'])/'sparkler_mpi'\nmpi_base.mkdir(parents=True,exist_ok=False)\nsessions=mpi_base/'sessions';sessions.mkdir()\nbacking=mpi_base/'backing';backing.mkdir()\nvader=mpi_base/'vader';vader.mkdir()\nenv.update(OMPI_MCA_orte_tmpdir_base=str(sessions),OMPI_MCA_shmem_mmap_relocate_backing_file='1',OMPI_MCA_shmem_mmap_backing_file_base_dir=str(backing),OMPI_MCA_btl_vader_backing_directory=str(vader))\n"+marker
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lk:
 fcntl.flock(lk,fcntl.LOCK_SH);ledger=load(S/'benchmarks/sparkler.json');ct=ledger['accepted_named_contract']
 for r in load(P/'plans.json')['rows']:
  m=r['model'];np=Path(r['plan']['path']);new=load(np);old=load(np.with_name('original_plan.json'));rec=new['sparkler_private_MPI_recovery'];parent=Path(rec['parent_plan']['path']);raw=gzip.decompress(parent.read_bytes());ck(json.loads(raw)==old and sha(parent)==rec['parent_plan']['sha256'],'fulloriginalcapturedplan:'+m);ck((P/'retained_immutable'/m/'plan.json').read_bytes()==raw,'raworiginalplanbyteidentity:'+m)
  ck(str(parent.parent.relative_to(W))==ledger['models'][m]['latest_profile_attempt']['profile_id'],'canonicalcurrentcapture:'+m)
  for a,b in [('configuration_id','configuration_id'),('raw_config_sha256','raw_config_sha256'),('reference_result_sha256','result_sha256')]:ck(new[a]==old[a]==ct[b],'canonicalcontract:'+m+':'+a)
  ck(sha(new['reference_result'])==ct['result_sha256']and sha(Path(new['reference_result']).with_name('config.json'))==ct['raw_config_sha256'],'currentresultconfigbytes:'+m)
  recovered={x['copy']:x for x in old['completion_retry_proposal']['patches']};oldbind=old['execution_bindings'];newbind=new['execution_bindings'];expectedbind={}
  for path,h in oldbind.items():
   if Path(path).exists():ck(sha(path)==h and newbind.get(path)==h,'existinghistoricalbinding:'+m+':'+path);expectedbind[path]=h
   else:
    aa=[x for x in rec['historical_missing_binding_aliases']if x['historical_missing_path']==path];ck(len(aa)==1,'missingaliasunique:'+m+':'+path);a=aa[0];dest=Path(a['immutable_alias']['path']);data=dest.read_bytes();ck(hashlib.sha256(data).hexdigest()==h==a['expected_sha256'],'missingaliasexactSHA:'+m+':'+path)
    if path in recovered:ck(data==recovered[path]['text'].encode(),'retainedfulltextaliasidentity:'+m+':'+path)
    else:
     original=Path('/tmp/hecbench-roofline-v100-20261020/batch008-host-overlays/sparkler')/m/'compiler_manifest.json';ck(data==original.read_bytes(),'008manifestfullbyteidentity:'+m)
    ck(dest.is_relative_to(P/'retained_immutable')and not Path(path).exists(),'neverrecreatehistoricalpath:'+m+':'+path);expectedbind[str(dest)]=h;aliases.append(a)
  newsc=Path(new['scratch_cwd'])/'profile_source';oldsc=old['scratch_cwd']
  for x in rec['patches']:
   name=Path(x['copy']).name;before=x['before_text'];after=x['after_text'];ck(Path(x['copy']).read_bytes()==after.encode()and sha(x['copy'])==x['copy_sha256'],'newsourcebytes:'+m+':'+name)
   if name=='study_backend.cpp':
    ck(before.encode()==(SRC/name).read_bytes()and hashlib.sha256(before.encode()).hexdigest()==oldbind[str(SRC/name)],'backendbeforecanonical:'+m)
    a='study_need(visible&&std::string(visible)=="6","GPU6-only visibility required")';b='study_need(visible&&std::string(visible)=="'+UUID+'","exact GPU6 UUID visibility required")';expected=before.replace(a,b).replace('#include "study_capture.h"','#include '+json.dumps(str(SRC/'study_capture.h')));inverse=after.replace(b,a).replace('#include '+json.dumps(str(SRC/'study_capture.h')),'#include "study_capture.h"')
   else:
    q=next(q for q in recovered.values()if Path(q['copy']).name==name);ck(before==q['text']and hashlib.sha256(before.encode()).hexdigest()==q['copy_sha256'],'sourcebefore024fulltext:'+m+':'+name);expected=before.replace(oldsc,new['scratch_cwd']);inverse=after
    if name=='study_launch.py':expected=expected.replace(marker,insert);inverse=inverse.replace(insert,marker)
    inverse=inverse.replace(new['scratch_cwd'],oldsc)
   ck(expected==after and inverse==before,'fullsourceforwardinverse:'+m+':'+name);expectedbind[x['copy']]=x['copy_sha256'];sources.append({'model':m,'name':name,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':sha(x['copy']),'fullforward_inverse_equal':True})
  # Reconstruct every allowed changed plan field, then compare complete object.
  expected=copy.deepcopy(old)
  for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']:expected[k]=relocate(old[k],oldsc,new['scratch_cwd'])
  expected['build_command']=[('/usr/bin/nvcc'if x=='nvcc'else str(newsc/'study_backend.cpp')if x==str(SRC/'study_backend.cpp')else x)for x in expected['build_command']];expected['profile_target_command'][0]='/usr/bin/python3'
  recipe=np.parent/'fresh_compiler_manifest.json';rr=load(recipe);ck(rr['compiler_command']==expected['build_command']and rr['compiler_executed']is False and rr['GPU_executed']is False,'freshrecipeexactcommand:'+m);expectedbind[str(recipe)]=sha(recipe)
  additions=[parent,Path(rec['parent_provenance']['path']),parent.parent/'manifest.json',(P/'prepare_private.py').resolve()]
  HPC=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/2025/comm_libs/12.9/hpcx/hpcx-2.22.1/ompi');additions +=[HPC/'lib/openmpi/mca_shmem_mmap.so',HPC/'lib/openmpi/mca_btl_vader.so',HPC/'lib/libopen-rte.so',HPC/'share/man/man1/mpirun.1',Path('/usr/bin/nvcc').resolve()]
  for path in additions:expectedbind[str(path)]=sha(path)
  ck(expectedbind==newbind,'exactallnewsourcebindings_noarbitraryalias:'+m)
  for path,h in newbind.items():ck(sha(path)==h,'allcurrentnewbindingSHA:'+m+':'+path)
  expected['execution_bindings']=expectedbind;expected['notes']=old['notes']+['Fresh privately proposed sourcebound Sparkler MPIpath/immutablemanifest/UUIDguard recovery; rootreview required, runtimefix unproved.'];expected['commands_executed']={'build':False,'GPU':False,'target':False};expected['sparkler_private_MPI_recovery']=rec;ck(expected==new,'completewhitelistedplanforwardequality:'+m)
  inv=copy.deepcopy(new)
  for k in list(inv):
   if k not in old:del inv[k]
  for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets','execution_bindings','notes','commands_executed']:inv[k]=copy.deepcopy(old[k])
  ck(inv==old,'completeoldplaninverse:'+m);models.append({'model':m,'plan':B(np),'original_plan_sha256':hashlib.sha256(raw).hexdigest(),'alias_count':len(rec['historical_missing_binding_aliases']),'source_patches':len(rec['patches']),'canonical_capture':ledger['models'][m]['latest_profile_attempt']['profile_id']})
 fcntl.flock(lk,fcntl.LOCK_UN)
ck(len(aliases)==14,'exact14aliascount');ck(len([a for a in aliases if Path(a['historical_missing_path']).name=='compiler_manifest.json'])==3,'exact3compileraliases')
cmd=load(P/'future_command.json');argv=cmd['argv'];ck(cmd['maximum_collector_invocations']==1 and argv[argv.index('--timeout')+1]=='3600'and'--retain-auxiliary'not in argv and'--skip-qualified-selected'in argv,'finite3600defaultcleanup');ck(argv[argv.index('--reserve-mib')+1]=='2048'and cmd['prospective_peak_free_requirement_bytes']==800070340+5*1024**3,'explicitstorageplanninggate');method=Path(argv[argv.index('--profiler-method-review')+1]);ck(sha(method)==argv[argv.index('--profiler-method-review-sha256')+1]=='65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca','explicit2025methodreview');ck('--selection-overrides'not in argv,'defaultselectorunchanged')
for e in load(P/'tool_bindings.json')['files']:ck(sha(e['path'])==e['sha256'],'currenttoolbinding:'+e['path'])
report={'status':'SOURCE_REVIEW_PASS_CONDITIONAL_ROOT_PREFLIGHT_REQUIRED','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'proposal_DELIVERY':B(D),'checks':len(checks),'models':models,'aliases':aliases,'sources':sources,'installed_MPI_evidence_verified':True,'full_forward_inverse_source_and_plan_equality':True,'GPU_MPI_build_import_test_execution':0,'runtime_recovery_proven':False,'prelaunch_requirements':['Root rehash exactdelivery/tool/source/planbindings under sharedauthority lock; canonical selectedcapture/result/config still join','Root enforce currentfree >=6168779460B immediately before sole3600call; planningmargin not exactpeakproof; C21ongoing2GiBreserve unchanged','Root coordinates GPU6exclusive collector lock and priorfrontier completion; no automatic retry','Actual all5checks/counter/manifests, originalguard/kernel/timer/sourcequalifications and rootaudit/admission needed afterlaunch'],'risks':['MPIrelocation supported by exact installedDSO parameterdescription, not demonstrated runtime successful','OtherMPIcomponents/NCUkernel replay may fail or remain unqualified; no fallback/globalSHMwrite permitted','Profiler2025 explicit distinct methodstratum from original2022 failures, no causalfix claim','Freshscratch binaries cleaned afterfullmanifest; later byte-levelrecertification unavailable','No launchcensus/timedphase/fullROI or bottleneck certification'],'check_labels':checks};(O/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':len(checks),'models':len(models),'aliases':len(aliases),'review_sha256':sha(O/'review.json')}))
