from pathlib import Path
import json,hashlib
O=Path(__file__).parent.resolve();P=O/'plans.json';d=json.loads(P.read_bytes());H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();W=Path('/home/imo/HeCBench_julia');collector=W/'tools/v100_roofline_collect.py';method=W/'ontology/studies/v100_roofline_537/profiler_methods/ncu2025_2_0_V100_sm70_raw_v1/review.json';expected='65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca';assert H(method)==expected
jobs=[]
for b,batch in [('testSNAP','batch067-testSNAP-observer-path'),('simpleMultiDevice','batch068-simpleMultiDevice-GPU6UUID')]:
 root=Path('/tmp/hecbench-roofline-v100-20261020')/batch
 for row in [r for r in d['plans']if r['benchmark']==b]:
  path=Path(row['plan']['path']);p=json.loads(path.read_bytes());cmd=p['build_command']
  if cmd:
   if cmd[0]=='nvcc':cmd[0]='/usr/bin/nvcc'
   elif cmd[0]=='g++':cmd[0]='/usr/bin/g++'
   assert Path(cmd[0]).exists();p['execution_bindings'][str(Path(cmd[0]).resolve())]=H(Path(cmd[0]).resolve());p['review']['narrow_recovery']['compiler_resolution']='Existing compiler absolute path required because approved profiler method pins PATH; no compiler/package/toolchain upgrade.'
  p['execution_bindings'][str(Path(__file__).resolve())]=H(__file__)
  path.write_text(json.dumps(p,indent=2)+'\n');row['plan']['sha256']=H(path)
  original=json.loads(path.with_name('original_plan.json').read_text());path.with_name('plan_diff.json').write_text(json.dumps({k:{'before':original.get(k),'after':v}for k,v in p.items()if original.get(k)!=v},indent=2)+'\n')
 jobs.append({'benchmark':b,'maximum_invocations':1,'argv':['/usr/bin/python3',str(collector),'--source-root',str(W),'--plan-root',str(root),'--benchmark',b,'--timeout','3600','--reserve-mib','2048','--retain-auxiliary','--skip-qualified-selected','--profiler-executable','/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Compute/ncu','--profiler-method-review',str(method),'--profiler-method-review-sha256',expected]})
P.write_text(json.dumps(d,indent=2)+'\n');(O/'future_commands.json').write_text(json.dumps({'root_only_not_executed':True,'maximum_collector_invocations':2,'serial_order':['testSNAP','simpleMultiDevice'],'jobs':jobs,'required_root_review':'Exact source-copy plans, current collector+method pins; root must authorize future launch and supervise finite once-only jobs. No controller created/launched.','GPU':'Only physical GPU6 exact UUID, no GPU5 query or branch','unchanged_selector':['--filter-mode','per-launch-config','--kernel-name-base','mangled','--kernel-id','::regex:.*:1'],'expected_matching_launches':None},indent=2)+'\n');(O/'tool_bindings.json').write_text(json.dumps({'files':[{'path':str(p),'sha256':H(p)}for p in [collector,W/'tools/v100_profiler_method.py',W/'tools/v100_roofline_classify.py',W/'tools/v100_profile_write_guard.py',method]],'no_execution':True},indent=2)+'\n');print('prepared',len(d['plans']),'plans',H(P))
