"""CPU-only source/metadata revision builder. Does not import or run the runner."""
from pathlib import Path
import json,hashlib,shutil,ast,datetime
P=Path(__file__).resolve().parent
R=Path('/home/imo/HeCBench_julia')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=P/'frozen_v100_roofline_classify.py';shutil.copyfile(R/'tools/v100_roofline_classify.py',parser)
# Literal-only extraction of reviewed metric definition; no collector import.
tree=ast.parse((R/'tools/v100_roofline_collect.py').read_text());node=next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='METRICS'for t in n.targets))
# This list is intentionally explicit and source-expression bound for root review.
metrics=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','dram__bytes.sum','dram__bytes.sum.peak_sustained','dram__cycles_elapsed.avg.per_second','sm__cycles_elapsed.avg.per_second','lts__t_bytes.sum','sm__inst_executed_pipe_tensor.sum','lts__t_bytes.sum.peak_sustained','lts__cycles_elapsed.avg.per_second','sm__inst_executed_pipe_tensor.sum.peak_sustained','sm__throughput.avg.pct_of_peak_sustained_elapsed','dram__throughput.avg.pct_of_peak_sustained_elapsed','smsp__sass_thread_inst_executed_op_integer_pred_on.sum']+['smsp__sass_thread_inst_executed_op_'+x+'_pred_on.sum'for x in ('hadd','hmul','hfma','fadd','fmul','ffma','dadd','dmul','dfma')]+['sm__sass_thread_inst_executed_op_'+x+'_pred_on.sum.peak_sustained'for x in ('hfma','ffma','dfma')]
units={}
for k in metrics:
 units[k]=None if k=='gpu__time_duration.sum'else 'cycle/second'if k.endswith('per_second')else 'byte/cycle'if k in ('dram__bytes.sum.peak_sustained','lts__t_bytes.sum.peak_sustained')else 'inst/cycle'if k.endswith('peak_sustained')else '%'if 'pct_of_peak'in k else 'byte'if k.startswith(('dram__bytes','lts__t_bytes'))else 'inst'
pci=['device__attribute_pci_'+k+'_id'for k in ('domain','bus','device')];cc=['device__attribute_compute_capability_major','device__attribute_compute_capability_minor']
source=P/'run_proposal.py';text=source.read_text().replace('import argparse,array,csv,','import argparse,array,csv,math,')
a=text.index("  expected=[]\n",text.index('def assert_activity'));b=text.index(' # No same-session',a)
text=text[:a]+'''  expected=plan['registered_executables']['profiler' if r.get('role')=='profiler'else 'worker']
  if live['exe']in expected:allowed[r['pid']]=r
'''+text[b:]
a="  if r.get('role')=='profiler'and app['gpu_uuid']!=snapshot['devices']['6']['uuid']:raise RuntimeError('profilerunexpectedGPU5context')"
b="""  physical='6'if r.get('role')=='profiler'or r.get('rank')==1 else '5'if r.get('rank')==0 else None
  if physical is None or app['gpu_uuid']!=snapshot['devices'][physical]['uuid']:raise RuntimeError('exactrankphysicalUUIDmappingrequired '+json.dumps(app))"""
text=text.replace(a,b)
a=text.index('def check_counter_rows(');b=text.index('\ndef validate_outputs(',a)
text=text[:a]+'''def check_counter_rows(slot,plan,snapshot,rank1):
 path=slot/'counters.csv';parser_path=Path(plan['counter_parser']['path'])
 if sha(parser_path)!=plan['counter_parser']['sha256']:raise RuntimeError('frozenparserSHA')
 parser=types.ModuleType('frozen_allreduce_counter_parser');exec(compile(parser_path.read_text(),str(parser_path),'exec'),parser.__dict__)
 groups,counter_sha=parser.read_csv(path)
 if len(groups)!=1:raise RuntimeError('exactoneisolatedaddcountergrouprequired')
 pci=snapshot['devices']['6']['pci'];domain,bus,dev=pci.split(':');expected=(int(domain,16),int(bus,16),int(dev.split('.')[0],16));out=[]
 for g in groups:
  ident=g['identity'];name=ident['Kernel Name'];metrics=g['metrics'];units=g['units']
  if not re.search(plan['kernel_regex'],name)or int(ident['Process ID'])!=rank1:raise RuntimeError('counterrank1/familymismatch')
  for key,want_unit in plan['required_metric_units'].items():
   value=metrics.get(key)
   if value is None or not math.isfinite(value)or value<0:raise RuntimeError('required27metric_missing_or_invalid '+key)
   if key=='gpu__time_duration.sum':
    if value<=0 or units.get(key)not in parser.TIME:raise RuntimeError('duration_positive_known_unit')
   elif units.get(key)!=want_unit:raise RuntimeError('required27metric_unit '+key+' '+str(units.get(key)))
  for key in plan['required_numeric_device_attributes']:
   value=metrics.get(key)
   if value is None or not math.isfinite(value)or value<0 or int(value)!=value or units.get(key)!='':raise RuntimeError('deviceattribute_numeric_dimensionless '+key)
  actual=tuple(metrics.get('device__attribute_pci_'+key+'_id')for key in ['domain','bus','device'])
  if actual!=expected:raise RuntimeError('rawcounterPCI_notphysicalGPU6')
  if(metrics.get('device__attribute_compute_capability_major'),metrics.get('device__attribute_compute_capability_minor'))!=(7,0):raise RuntimeError('rawcounter_sm70_required')
  for key,want in [('Grid Size',[32,1,1]),('Block Size',[256,1,1])]:
   values=[int(v)for v in re.findall(r'\\d+',ident[key]or'')]
   if values!=want:raise RuntimeError('exact32x256geometryrequired')
  # Preserve all automatic rows and units; parser intentionally maps string
  # attributes to None. Original raw CSV remains immutable evidence.
  out.append(g)
 return {'counter_sha256':counter_sha,'parser_sha256':sha(parser_path),'groups':out,'required_standard_metric_count':27,'required_numeric_identity_attributes':plan['required_numeric_device_attributes'],'physical_GPU6_UUID':snapshot['devices']['6']['uuid'],'physical_GPU6_PCI':expected,'rank1_worker_pid':rank1,'gpu_sm70':True,'scope':'One retainedtrial2 size1M firstrepeat rank1 privateadd only; not collective/wholeROI.'}

def wait_both_idle(bundle,deadline,receipt):
 while True:
  snap=device_snapshot(bundle);receipt.append(snap)
  if not snap['compute_processes']:return snap
  if time.monotonic()>=deadline:raise RuntimeError('bounded_both_device_idle_wait_expired_no_signals')
  time.sleep(min(5,max(0,deadline-time.monotonic())))
''' +text[b:]
text=text.replace("p.add_argument('--authorization-sha256');a=p.parse_args();", "p.add_argument('--authorization-sha256');p.add_argument('--queue-wait-seconds',type=int,default=86400);a=p.parse_args();")
text=text.replace(" verify(bundle['all_readonly_bindings']);authorization=", " if not 1<=a.queue_wait_seconds<=86400:raise RuntimeError('queuewaitmustbein1to86400')\n verify(bundle['all_readonly_bindings']);authorization=")
text=text.replace("'deadline_seconds_per_model':1800}","'deadline_seconds_per_model':1800,'maximum_queue_wait_seconds':86400}")
text=text.replace("until=time.monotonic()+60", "until=time.monotonic()+a.queue_wait_seconds;queue={'started_at':utc(),'maximum_wait_seconds':a.queue_wait_seconds,'observer_budget_excludes_queue_wait':True,'device_idle_snapshots':[]};dump(out/'queue_wait_receipt.json',queue)")
text=text.replace("  verify(bundle['all_readonly_bindings'])\n  for m", "  wait_both_idle(bundle,until,queue['device_idle_snapshots']);queue['finished_at']=utc();dump(out/'queue_wait_receipt.json',queue)\n  verify(bundle['all_readonly_bindings'])\n  for m")
# Every per-model fresh idle wait uses own finite queue window; not observer budget.
text=text.replace("   snap=device_snapshot(bundle)\n   if snap['compute_processes']:raise RuntimeError('bothGPUsmustbeidle_defer')", "   idle_receipt=[];snap=wait_both_idle(bundle,time.monotonic()+a.queue_wait_seconds,idle_receipt);dump(slot/'idle_wait_receipt.json',{'snapshots':idle_receipt,'observer_budget_excludes_wait':True,'maximum_wait_seconds':a.queue_wait_seconds});verify(bundle['all_readonly_bindings'])")
text=text.replace(" finally:\n  for fd in reversed(fds)"," finally:\n  if 'queue'in locals():dump(out/'queue_wait_receipt.json',queue)\n  for fd in reversed(fds)")
source.write_text(text)
d=json.loads((P/'proposal.json').read_text());d['revision']=2;d['revises_frozen_v1']='v1_frozen/FREEZE_MANIFEST.json';d['created_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
wrapper=Path('/usr/lib/nsight-compute/ncu');elf=wrapper.parent/'target/linux-desktop-glibc_2_11_3-x64/ncu'
patterns={'cuda':r'^(?:void )?kernel_add\(','cuda_jl':r'^add_kernel_\(','ka':r'^gpu_add_kernel_\(','jacc':r'^_parallel_for_cuda\([^\n]*\badd_kernel_(?=[,<(])'}
for model,rec in d['models'].items():
 rec['kernel_regex']=patterns[model];args=rec['ncu_arguments'];args[args.index('--kernel-name')+1]='regex:'+patterns[model];args[args.index('--metrics')+1]=','.join(metrics+pci+cc)
 rec['ncu_wrapper_source']={'path':str(wrapper),'sha256':sha(wrapper)};rec['ncu_executable']=str(elf);rec['ncu_executable_sha256']=sha(elf)
 rec['registered_executables']={'profiler':[str(elf)],'worker':['{scratch}/study_native']if model=='cuda'else [str(Path('/home/imo/.juliaup/bin/julia').resolve()),'/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia']}
 rec['counter_parser']={'path':str(parser),'sha256':sha(parser)};rec['required_standard_metrics']=metrics;rec['required_metric_units']=units;rec['required_numeric_device_attributes']=pci+cc
 rec['execution_bindings'].update({str(parser):sha(parser),str(wrapper):sha(wrapper),str(elf):sha(elf),str(source):sha(source)})
 for path in list(rec['execution_bindings']):
  if str(P)in path and Path(path).is_file():rec['execution_bindings'][path]=sha(path)
d['counter_contract']={'standard_metric_count':27,'collector_metric_expression':ast.get_source_segment((R/'tools/v100_roofline_collect.py').read_text(),node),'collector_path':str(R/'tools/v100_roofline_collect.py'),'collector_sha256':sha(R/'tools/v100_roofline_collect.py'),'identity_attributes':pci+cc,'requested_metric_count':32,'all27finite_nonnegative_known_units_required':True,'PCI_and_sm70_raw_observation_required':True,'nonnumeric_automatic_attributes':'preserved as None by exact frozen parser; raw CSV retained'}
d['queue_policy']={'maximum_configurable_wait_seconds':86400,'outside_per_model_observer_deadline':True,'both_locks_and_both_device_idle_required':True,'no_foreign_signals':True}
d['ncu_direct_ELF_adaptation']={'wrapper':str(wrapper),'wrapper_sha256':sha(wrapper),'target':str(elf),'target_sha256':sha(elf),'proof':'Exact x86_64 wrapper branch invokes this ELF with unchanged argv but without exec. Invoke that exact ELF directly so registered profiler PID survives exec; no wildcard or child PID grace. Host launcher adaptation requires root review; unvalidated.','host_architecture_required':'x86_64'}
for path in list(d['all_readonly_bindings']):
 if str(P)in path and Path(path).is_file():d['all_readonly_bindings'][path]=sha(path)
for path in [parser,wrapper,elf,R/'tools/v100_roofline_collect.py']:d['all_readonly_bindings'][str(path)]=sha(path)
d['runner']['sha256']=sha(source);d['runner']['bytes']=source.stat().st_size
d['preparation_corrections'].append('V2: frozen root-reviewed v1 preserved; exact standard27+3PCI+2CC requested and checked by frozen parser; single-escape Python/NCU regex; rank0GPU5/rank1GPU6; exact ELF instead of fork-producing shell wrapper; configurable queue/idle wait up to86400 outside observer budget. No runner execution/tests/build/GPU queries.')
(P/'proposal.json').write_text(json.dumps(d,indent=2)+'\n')
a=json.loads((P/'authorization_template.json').read_text());a['maximum_queue_wait_seconds']=86400;(P/'authorization_template.json').write_text(json.dumps(a,indent=2)+'\n')
