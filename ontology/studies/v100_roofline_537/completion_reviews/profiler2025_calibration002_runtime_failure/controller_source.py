#!/usr/bin/env python3
"""Root-reviewed NCU2025 method calibration; no build, ledger or adoption changes.
Without --execute, validates arguments/bindings only and never runs any process.
Prepared by source inspection only; author did not execute this controller.
"""
import argparse,ast,csv,datetime,fcntl,hashlib,json,math,os,re,sys,time,types
from pathlib import Path
COLLECTOR_SHA='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c'
UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.next');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def verify(bindings):
 for p,h in bindings.items():
  if sha(p)!=h:raise RuntimeError('Prospectively bound source/tool/ELF/reference changed: '+p)
 return{'checked_at':utc(),'bindings':bindings,'all_unchanged':True}
def module(path,name):
 obj=types.ModuleType(name);obj.__file__=str(path);exec(compile(Path(path).read_bytes(),str(path),'exec'),obj.__dict__);return obj
def guarded_payload(path,expected_write_directories):
 """Exact one-prefix normalization; full merged log remains unchanged.
 Only the source-defined LLguard line is removed, never arbitrary warnings.
 """
 path=Path(path);raw=path.read_bytes();prefix,separator,payload=raw.partition(b'\n')
 if not separator or not prefix.startswith(b'V100_WRITE_GUARD '):raise RuntimeError('Exact LLguard preface missing '+str(path))
 guard=json.loads(prefix[len(b'V100_WRITE_GUARD '):])
 if set(guard)!={'landlock_abi','write_directories','privilege'}:raise RuntimeError('Unexpected LLguard schema')
 if type(guard['landlock_abi'])is not int or guard['landlock_abi']<3:raise RuntimeError('LLguard ABI mismatch')
 if guard['write_directories']!=expected_write_directories:raise RuntimeError('LLguard write directory differs from actual freshoutput')
 if guard['privilege']!='existing unprivileged user; no_new_privs':raise RuntimeError('LLguard privilege mismatch')
 if not payload or any(line.startswith(b'V100_WRITE_GUARD ')for line in payload.splitlines()):raise RuntimeError('Empty or repeated LLguard payload')
 normalized=path.with_name(path.stem+'.payload.bin');normalized.write_bytes(payload)
 receipt={'raw_log':str(path),'raw_log_sha256':sha(path),'raw_log_bytes':len(raw),
          'exact_guard_preface_sha256':hashlib.sha256(prefix+b'\n').hexdigest(),
          'normalized_payload':str(normalized),'normalized_payload_sha256':sha(normalized),'normalized_payload_bytes':len(payload),
          'normalization':'Remove exactly one first source-defined LLguard JSON line; preserve every remaining byte; merged stdout/stderr not retroactively separated',
          'guard':guard}
 return payload,receipt

def shape(value):
 n=[int(x)for x in re.findall(r'\d+',value or'')]
 if len(n)!=3:raise RuntimeError('Unrecognized launch shape '+repr(value))
 return tuple(n)
def key(g):
 name=g['identity']['Kernel Name'];family=None
 for f in ['calibration_half2','calibration_half','calibration_tensor','calibration_hadd2','calibration_hmul2']:
  if re.search(r'\b'+f+r'\(',name):family=f;break
 if 'selector_typed<float>'in name:family='selector_float'
 if 'selector_typed<double>'in name:family='selector_double'
 if family is None:raise RuntimeError('Unknown calibration function identity '+name)
 return(family,shape(g['identity']['Block Size']),shape(g['identity']['Grid Size']))
def rows_by_key(groups):
 result={}
 for row in groups:
  k=key(row)
  if k in result:raise RuntimeError('Duplicate function+launch geometry '+repr(k))
  result[k]=row
 return result
def instr(s):return'smsp__sass_thread_inst_executed_op_'+s+'_pred_on.sum'
def positive_number(x):return isinstance(x,(int,float))and not isinstance(x,bool)and math.isfinite(x)and x>0
def validate(case,groups,old,metrics,expected_pci):
 expected={
 ('calibration_half2',(128,1,1),(32,1,1)):(instr('hfma'),524288,2097152,2097152),
 ('calibration_half',(128,1,1),(32,1,1)):(instr('hfma'),524288,2097152,1048576),
 ('calibration_tensor',(32,1,1),(1,1,1)):('sm__inst_executed_pipe_tensor.sum',2048,1048576,1048576),
 ('selector_float',(128,1,1),(32,1,1)):(instr('ffma'),4096,8192,8192),
 ('selector_double',(128,1,1),(32,1,1)):(instr('dfma'),4096,8192,8192),
 ('selector_float',(256,1,1),(16,1,1)):(instr('ffma'),4096,8192,8192),
 ('selector_double',(256,1,1),(16,1,1)):(instr('dfma'),4096,8192,8192),
 ('calibration_hadd2',(128,1,1),(32,1,1)):(instr('hadd'),524288,1048576,1048576),
 ('calibration_hmul2',(128,1,1),(32,1,1)):(instr('hmul'),524288,1048576,1048576)}
 expected={k:v for k,v in expected.items()if(case=='precision')==(k[0]not in ['calibration_hadd2','calibration_hmul2'])}
 actual=rows_by_key(groups);oldrows=rows_by_key(old)
 if set(actual)!=set(expected)or set(oldrows)!=set(expected):raise RuntimeError('Need exactly7precision+2arithmetic identities/geometries; no duplicate/missing rows')
 details=[]
 for k,g in actual.items():
  mm=g['metrics'];units=g['units'];missing=[x for x in metrics if x not in mm or mm[x]is None or not isinstance(mm[x],(int,float))or not math.isfinite(mm[x])or mm[x]<0]
  if missing:raise RuntimeError('Requested full collector.METRICS missing/nonfinite/unsupported: '+repr(missing))
  pci=tuple(mm.get('device__attribute_pci_'+x+'_id')for x in ['domain','bus','device'])
  if pci!=expected_pci:raise RuntimeError('Actual raw counter PCI mismatch '+repr(pci))
  if (mm.get('device__attribute_compute_capability_major'),mm.get('device__attribute_compute_capability_minor'))!=(7,0):raise RuntimeError('Raw counters are notsm70')
  if mm['gpu__time_duration.sum']<=0:raise RuntimeError('Missing positive duration')
  for m in metrics:
   u=units.get(m)
   if not u:raise RuntimeError('Explicit metric units missing '+m)
   expectedunit=('inst'if m.startswith(('smsp__sass_thread_inst_executed_op_','sm__inst_executed_pipe_tensor.sum'))and not m.endswith('peak_sustained')else 'byte'if m in ['dram__bytes_read.sum','dram__bytes_write.sum','dram__bytes.sum','lts__t_bytes.sum']else 'cycle/second'if m.endswith('avg.per_second')else '%'if m.endswith('pct_of_peak_sustained_elapsed')else 'byte/cycle'if m in ['dram__bytes.sum.peak_sustained','lts__t_bytes.sum.peak_sustained']else 'inst/cycle'if m.endswith('peak_sustained')else None)
   if m=='gpu__time_duration.sum':
    if u not in ['nsecond','ns','usecond','us','msecond','ms','second','s']:raise RuntimeError('Unrecognized duration unit '+u)
   elif expectedunit and u!=expectedunit:raise RuntimeError('Metric unit mismatch '+m+' '+str(u)+' expected '+expectedunit)
  metric,count,physical,useful=expected[k]
  if mm[metric]!=count:raise RuntimeError('Source-known typed instruction count mismatch '+repr((k,metric,mm[metric],count)))
  compared={}
  for m,v in oldrows[k]['metrics'].items():
   # Strict invariant comparison of historically measured typed instructions and
   # architectural instruction ceilings only. Timing/clocks/traffic may differ.
   if m.startswith('smsp__sass_thread_inst_executed_op_')or m in ['sm__inst_executed_pipe_tensor.sum','sm__inst_executed_pipe_tensor.sum.peak_sustained','sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained']:
    if v is None or mm.get(m)!=v:raise RuntimeError('Old2022 typed/architectural count differs '+repr((k,m,v,mm.get(m))))
    if oldrows[k]['units'].get(m)!=units.get(m):raise RuntimeError('Old/new invariant metric units differ '+m)
    compared[m]={'old2022':v,'new2025':mm[m]}
  details.append({'identity':g['identity'],'source_family':k[0],'expected_count':count,'count_metric':metric,'actual_count':mm[metric],'physical_executed_FLOPs':physical,'useful_source_FLOPs':useful,'physical_useful_distinction':'Scalarhalf still executes HFMA2 packed2lane×2ops; source uses only one lane. Never change globalphysical conversion to1lane.','old2022_invariant_counter_comparisons':compared,'actual_PCI':pci,'all_requested_metrics_present':True})
 return details

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--raw-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--timeout',type=int,default=600);ap.add_argument('--reserve-mib',type=int,default=2048);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 root=a.source_root.resolve();raw=a.raw_root.resolve();out=a.output_root.resolve()
 if a.output_root.is_symlink()or out.exists()or out==raw or not out.is_relative_to(raw)or out.name.startswith('batch'):raise RuntimeError('Fresh private non-batch output directory under ownedraw required')
 if not raw.is_relative_to('/tmp')or raw==Path('/tmp')or not raw.is_dir()or raw.stat().st_uid!=os.getuid():raise RuntimeError('Owned existing/tmp raw tree required')
 if not 1<=a.timeout<=3600 or a.reserve_mib<2048:raise RuntimeError('Bounded wall deadline and>=2GiB free reserve required')
 if sha(a.manifest)!=a.manifest_sha256:raise RuntimeError('Manifest SHA mismatch')
 manifest=json.loads(a.manifest.read_text());bound=manifest['bindings'];verify(bound)
 if manifest['collector_sha256']!=COLLECTOR_SHA or sha(root/'tools/v100_roofline_collect.py')!=COLLECTOR_SHA:raise RuntimeError('Only exact reviewed collector32a9 accepted')
 if str(root)!=manifest['source_root']or str(raw)!=manifest['raw_root']:raise RuntimeError('Source/raw root differs from manifest')
 if not a.execute:
  print(json.dumps({'status':'source_manifest_validated_only','GPU_build_processes':0,'no_output_created':True}));return 0
 out.mkdir();prov={'schema_version':1,'started_at':utc(),'status':'calibration_prepared','activity':'separate_profiler_version_method_validation_not_translation_or_unit_test','manifest':{'path':str(a.manifest.resolve()),'sha256':a.manifest_sha256},'source_tool_ELF_before':verify(bound),'cases':{},'build_processes':0,'no_ledger_or_method_adoption':True,'original_compiler_identity':'Oldbuildargvonly; actual original compiler executablehash/version notcaptured. Reusing exact existingbinary; no retrocertification.','gpu_uuid':UUID,'foreign_wait_policy':'Unbounded collector wait_idle_snapshot, outside per-process operational wall deadline. No foreign signals.'}
 dump(out/'profile_provenance.json',prov)
 collector=module(root/'tools/v100_roofline_collect.py','pinned_collector32a9');parser=module(root/'tools/v100_roofline_classify.py','pinned_calibration_parser')
 if collector.COLLECTOR_SHA256!=COLLECTOR_SHA or collector.UUID!=UUID:raise RuntimeError('Imported collector identity differs')
 metrics=list(collector.METRICS)
 if len(metrics)!=len(set(metrics)):raise RuntimeError('Duplicate requested metrics')
 prov['requested_metrics']=metrics;prov['collector_source_sha256']=collector.COLLECTOR_SHA256
 (out/'collector_source.py').write_bytes(collector.COLLECTOR_SOURCE);(out/'parser_source.py').write_bytes((root/'tools/v100_roofline_classify.py').read_bytes());(out/'controller_source.py').write_bytes(Path(__file__).read_bytes())
 ncu=manifest['ncu_executable'];guard={'tool':str(root/'tools/v100_profile_write_guard.py'),'sha256':sha(root/'tools/v100_profile_write_guard.py'),'write_directories':[str(out)],'policy':'Exact existingchildLandlock guard; onlyfreshmethodoutput writable, privateCUDA/XDG/TMPcaches; no unguardedfallback orsharedcaches'}
 prov['filesystem_write_guard']=guard;allrows=[]
 try:
  with Path('/tmp/hecbench-julia-study-gpu.lock').open('rb')as lock:
   began=time.monotonic();fcntl.flock(lock,fcntl.LOCK_EX);prov['common_lock_wait_seconds']=time.monotonic()-began;verify(bound)
   for case,paths in manifest['cases'].items():
    slot=out/case;slot.mkdir()
    for name in ['ncu_tmp','cuda_cache','xdg_cache']:(slot/name).mkdir()
    env=os.environ.copy();env.update(manifest['pinned_runtime_environment']);env.update(CUDA_VISIBLE_DEVICES=UUID,TMPDIR=str(slot/'ncu_tmp'),CUDA_CACHE_PATH=str(slot/'cuda_cache'),XDG_CACHE_HOME=str(slot/'xdg_cache'),PYTHONDONTWRITEBYTECODE='1')
    if Path(__import__('shutil').which('nvidia-smi',path=env.get('PATH'))or'').resolve()!=Path(manifest['nvidia_smi_executable']).resolve():raise RuntimeError('CollectorinternalNVMLqueryexecutable differs from boundpath')
    for k,v in manifest['pinned_inherited_environment'].items():
     if env.get(k,'')!=v:raise RuntimeError('Inheritedlibrary/injectionenvironment drift '+k)
    cp={'status':'prepared','source_bindings_before':verify(bound),'operational_deadline_seconds':a.timeout,'reserve_mib':a.reserve_mib};prov['cases'][case]=cp;dump(out/'profile_provenance.json',prov)
    waited=time.monotonic();cp['device_before']=collector.wait_idle_snapshot(prov,out);cp['idle_wait_seconds']=time.monotonic()-waited
    if __import__('shutil').disk_usage(out).free<a.reserve_mib*1024**2:raise RuntimeError('Reserve before calibration insufficient')
    # UUID/PCI query only physicalGPU6, guarded and foreign monitored like target.
    cp['identity_query']=collector.run([manifest['nvidia_smi_executable'],'-i','6','--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],env,slot,slot/'identity.log',60,a.reserve_mib*1024**2,guard,monitor_gpu=True)
    if cp['identity_query']['returncode']!=0 or cp['identity_query']['stop_reason']is not None:raise RuntimeError('Identity query process failed')
    identity_payload,identity_receipt=guarded_payload(slot/'identity.log',guard['write_directories'])
    cp['identity_output_normalization']=identity_receipt;dump(out/'profile_provenance.json',prov)
    identity_text=identity_payload.decode('utf-8')
    if not identity_payload.endswith(b'\n') or len(identity_text.splitlines())!=1:raise RuntimeError('Exactly one identity payload row required')
    ident=list(csv.reader(identity_text.splitlines()))
    if len(ident)!=1 or len(ident[0])!=2 or ident[0][0].strip()!=UUID:raise RuntimeError('GPU6 identity differs')
    domain,bus,device=ident[0][1].strip().split(':');pci=(int(domain,16),int(bus,16),int(device.split('.')[0],16));cp['expected_PCI']=pci
    if case=='precision':
     cp['profiler_version_process']=collector.run([ncu,'--version'],env,slot,slot/'version.log',60,a.reserve_mib*1024**2,guard,monitor_gpu=True)
     if cp['profiler_version_process']['returncode']!=0 or cp['profiler_version_process']['stop_reason']is not None:raise RuntimeError('Profiler runtimeversion process failed')
     version_payload,version_receipt=guarded_payload(slot/'version.log',guard['write_directories'])
     cp['version_output_normalization']=version_receipt;dump(out/'profile_provenance.json',prov)
     text=version_payload.decode('utf-8');version_lines=text.splitlines()
     if not version_payload.endswith(b'\n') or len(version_lines)!=3:raise RuntimeError('Unexpected profiler version payload shape')
     if version_lines[0]!='NVIDIA (R) Nsight Compute Command Line Profiler':raise RuntimeError('Unexpected profiler header')
     if re.fullmatch(r'Copyright \(c\) 2018-20[0-9]{2} NVIDIA Corporation',version_lines[1])is None:raise RuntimeError('Unexpected profiler copyright')
     if re.fullmatch(r'Version 2025\.2\.0\.0 \(build [0-9]+\) \(public-release\)',version_lines[2])is None:raise RuntimeError('Unexpected exact profiler version/build')
     prov['actual_profiler_version_text']=(slot/'version.log').read_text()
     prov['actual_profiler_version_payload_text']=text
     prov['actual_profiler_version_payload_sha256']=version_receipt['normalized_payload_sha256']
    command=[ncu,'--target-processes','all','--filter-mode','per-launch-config','--kernel-name-base','mangled','--kernel-id','::regex:.*:1','--clock-control','none','--cache-control','all','--replay-mode','kernel','--metrics',','.join(metrics),'--csv','--page','raw','--print-units','base','--log-file',str(slot/'counters.csv'),paths['binary']]
    cp['command']=command;dump(out/'profile_provenance.json',prov)
    cp['profile']=collector.run(command,env,slot,slot/'execution.log',a.timeout,a.reserve_mib*1024**2,guard,monitor_gpu=True)
    # collector.run returns only after own-session quiescence; no scratch cleanup.
    cp['source_tool_ELF_after']=verify(bound)
    if (slot/'counters.csv').exists():cp['counters_sha256']=sha(slot/'counters.csv')
    dump(out/'profile_provenance.json',prov)
    if cp['profile']['returncode']!=0 or cp['profile']['stop_reason']is not None:raise RuntimeError('Profiler execution failed; raw logs/counters preserved; no dropped metrics')
    lines=[x for x in (slot/'execution.log').read_text(errors='replace').splitlines()if x.startswith('calibration_correct=')]
    if len(lines)!=1 or lines[0]!=paths['expected_host_check_line']:raise RuntimeError('Full original hostcalibration_correct line missing/mismatch')
    cp['full_host_check']={'line':lines[0],'pass':True,'scope':'Immutable originalcanaryfullarrayhostchecks, not merely scalarreturnedstatus'}
    oldgroups,oldsha=parser.read_csv(Path(paths['old_counters']));groups,newsha=parser.read_csv(slot/'counters.csv');cp['counter_comparisons']=validate(case,groups,oldgroups,metrics,pci);cp['old_counter_sha256']=oldsha;cp['new_counter_sha256']=newsha
    allrows.extend(cp['counter_comparisons']);cp['status']='calibration_case_pass';dump(out/'profile_provenance.json',prov)
   if len(allrows)!=9:raise RuntimeError('Notallninecalibrationrowsvalidated')
   prov['source_tool_ELF_final']=verify(bound);prov.update(status='new_profiler_calibration_pass_root_adoption_still_required',finished_at=utc(),calibration_rows=allrows,physical_half_lane_factor=2,physical_half_fma_ops_per_instruction=4,tensor_FLOPs_per_HMMA884_instruction=512,calibrated_scope='V100sm70existingbinariesonly; noclaimotherarchitectures/precisions, fullROI or causalbottleneck',timing_comparability='Durations/clocks/trafficnotrequiredidentical; oldandnewtoolstrataremainseparate',registry_or_method_adopted=False)
 except Exception as error:
  prov.update(status='new_profiler_calibration_failed_no_adoption',error=repr(error),finished_at=utc())
  try:prov['source_tool_ELF_final']=verify(bound)
  except Exception as binding_error:prov['final_binding_error']=repr(binding_error)
  dump(out/'profile_provenance.json',prov);print(json.dumps({'status':prov['status'],'error':repr(error),'output':str(out)}),flush=True);return 1
 dump(out/'profile_provenance.json',prov);print(json.dumps({'status':prov['status'],'rows':len(allrows),'output':str(out)}),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
