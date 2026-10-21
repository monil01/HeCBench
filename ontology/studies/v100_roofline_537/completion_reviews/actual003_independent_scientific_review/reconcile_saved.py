from pathlib import Path
import json,csv,io,hashlib,math,re,ast,datetime
R=Path('/tmp/hecbench-roofline-v100-20261020/method_calibration_ncu2025_2_0_003');V=Path('/tmp/roofline-completion-20261021/profiler2025_method_patch/v5');O=Path(__file__).resolve().parent
inputs={}
def read(p):
 p=Path(p);b=p.read_bytes();assert str(p)not in inputs or inputs[str(p)]==b;inputs[str(p)]=b;return b
def sha(b):return hashlib.sha256(b).hexdigest()
def hashfile(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def need(v,s):
 if not v:raise ValueError(s)
def shape(s):
 a=tuple(map(int,re.findall(r'\d+',s)));need(len(a)==3,'geometry');return a
def num(v):
 try:x=float(v.replace(',',''))
 except (ValueError,AttributeError):return None
 return x if math.isfinite(x)and x>=0 else None
def table(path):
 raw=read(path);lines=raw.decode().splitlines();starts=[i for i,x in enumerate(lines)if x.startswith('"ID"')];need(len(starts)==1,'header count');rows=list(csv.reader(io.StringIO('\n'.join(lines[starts[0]:]))));hdr=rows[0];need(len(hdr)==len(set(hdr)),'duplicate header');need(rows[1][hdr.index('ID')]=='','unit row missing');units=dict(zip(hdr,rows[1]));result={}
 for vals in rows[2:]:
  need(len(vals)==len(hdr),'row width');row=dict(zip(hdr,vals));name=row['Kernel Name'];family=next((x for x in ['calibration_half2','calibration_half','calibration_tensor','calibration_hadd2','calibration_hmul2']if re.match(re.escape(x)+r'\(',name)),None)
  if name.startswith('void selector_typed<float>'):family='selector_float'
  if name.startswith('void selector_typed<double>'):family='selector_double'
  need(family is not None,'unknown family '+name);key=(family,shape(row['Block Size']),shape(row['Grid Size']));need(key not in result,'duplicate geometry/family');result[key]={'row':row,'numeric':{k:num(v)for k,v in row.items()if '__'in k},'units':units}
 return result,sha(raw),[x for x in lines[:starts[0]]if x.startswith('==WARNING==')]
def guard_payload(path,receipt,slot):
 raw=read(path);first,sep,payload=raw.partition(b'\n');need(sep and first.startswith(b'V100_WRITE_GUARD '),'guard prefix');g=json.loads(first[len(b'V100_WRITE_GUARD '):]);need(set(g)=={'landlock_abi','write_directories','privilege'}and type(g['landlock_abi'])is int and g['landlock_abi']>=3,'guard schema/ABI');need(g['write_directories']==[str(R)]and g['privilege']=='existing unprivileged user; no_new_privs','guard scope/privilege');need(not any(x.startswith(b'V100_WRITE_GUARD ')for x in payload.splitlines()),'extra guard');need(sha(raw)==receipt['raw_log_sha256']and len(raw)==receipt['raw_log_bytes'],'normalization raw receipt');need(sha(first+b'\n')==receipt['exact_guard_preface_sha256']and sha(payload)==receipt['normalized_payload_sha256']and len(payload)==receipt['normalized_payload_bytes'],'normalization receipt');need(payload==read(slot),'payload bytes');return payload,g
praw=read(R/'profile_provenance.json');p=json.loads(praw);mraw=read(p['manifest']['path']);m=json.loads(mraw);need(sha(mraw)==p['manifest']['sha256'],'manifest receipt');need(len(m['bindings'])==83,'83binding count');need(set(p['cases'])==set(m['cases'])=={'precision','half_arithmetic'},'two casegroups')
helper=read(V/'proposed/tools/v100_profiler_method.py');tree=ast.parse(helper);constants={x.targets[0].id:ast.literal_eval(x.value)for x in tree.body if isinstance(x,ast.Assign)and len(x.targets)==1 and isinstance(x.targets[0],ast.Name)and x.targets[0].id in ['METHOD','UUID','CONTROLLER','MANIFEST']}
need(constants['MANIFEST']==sha(mraw)and constants['CONTROLLER']==sha(read(R/'controller_source.py')),'v5 exact experiment/controller');need(constants['UUID']==p['gpu_uuid']=='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d','UUID');need(p['status']=='new_profiler_calibration_pass_root_adoption_still_required'and p['registry_or_method_adopted']is False and p['no_ledger_or_method_adoption']is True,'adoption scope');need(p['build_processes']==0,'no rebuild');need(m['pinned_runtime_environment']=={'PATH':'/usr/bin:/bin','LD_LIBRARY_PATH':'','LD_PRELOAD':''}and m['pinned_inherited_environment']=={},'pinned runtime')
need(p['source_tool_ELF_before']['bindings']==p['source_tool_ELF_final']['bindings']==m['bindings'],'global recorded binding equality');need(p['source_tool_ELF_before']['all_unchanged']is True and p['source_tool_ELF_final']['all_unchanged']is True,'global binding flags')
bindings=[]
for name,h in m['bindings'].items():
 actual=hashfile(name);need(actual==h,'current83 binding differs '+name);bindings.append({'path':name,'expected_sha256':h,'current_sha256':actual})
for name,path in [('collector_source.py',Path(m['source_root'])/'tools/v100_roofline_collect.py'),('parser_source.py',Path(m['source_root'])/'tools/v100_roofline_classify.py')]:need(sha(read(R/name))==m['bindings'][str(path)],'frozen input source '+name)
for name,path in [('calibration_collector_source.py','tools/v100_roofline_collect.py'),('calibration_write_guard_source.py','tools/v100_profile_write_guard.py')]:need(sha(read(V/'calibration_input_sources'/name))==m['bindings'][str(Path(m['source_root'])/path)],'v5 input archive '+name)
v5inventory=json.loads(read(V/'authoring_bindings.json'));candidatefiles=[]
for x in v5inventory:
 q=V/'proposed'/x['file'];need(sha(read(q))==x['v5_sha256'],'v5 candidate inventory '+str(q));candidatefiles.append({'path':str(q),'sha256':x['v5_sha256']})
metrics=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','dram__bytes.sum','dram__bytes.sum.peak_sustained','dram__cycles_elapsed.avg.per_second','sm__cycles_elapsed.avg.per_second','lts__t_bytes.sum','sm__inst_executed_pipe_tensor.sum','lts__t_bytes.sum.peak_sustained','lts__cycles_elapsed.avg.per_second','sm__inst_executed_pipe_tensor.sum.peak_sustained','sm__throughput.avg.pct_of_peak_sustained_elapsed','dram__throughput.avg.pct_of_peak_sustained_elapsed','smsp__sass_thread_inst_executed_op_integer_pred_on.sum']+[f'smsp__sass_thread_inst_executed_op_{x}_pred_on.sum'for x in ['hadd','hmul','hfma','fadd','fmul','ffma','dadd','dmul','dfma']]+[f'sm__sass_thread_inst_executed_op_{x}_pred_on.sum.peak_sustained'for x in ['hfma','ffma','dfma']]
need(p['requested_metrics']==metrics and len(metrics)==27,'complete exact metric list');collecttree=ast.parse(read(R/'collector_source.py'));metricexpr=next(x.value for x in collecttree.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='METRICS'for t in x.targets));need(ast.dump(metricexpr)==ast.dump(ast.parse(m['metrics_source_expression'],mode='eval').body),'manifest metric expression matches original collector')
freq=('dram__cycles_elapsed.avg.per_second','sm__cycles_elapsed.avg.per_second','lts__cycles_elapsed.avg.per_second');instr=lambda s:f'smsp__sass_thread_inst_executed_op_{s}_pred_on.sum'
expected={('calibration_half2',(128,1,1),(32,1,1)):(instr('hfma'),4096*128,4*4096*128,4*4096*128),('calibration_half',(128,1,1),(32,1,1)):(instr('hfma'),4096*128,4*4096*128,2*4096*128),('calibration_tensor',(32,1,1),(1,1,1)):('sm__inst_executed_pipe_tensor.sum',2048,2*16**3*128,2*16**3*128)}
for b,g in [(128,32),(256,16)]:
 for fam,op in [('selector_float','ffma'),('selector_double','dfma')]:expected[(fam,(b,1,1),(g,1,1))]=(instr(op),4096,8192,8192)
for fam,op in [('calibration_hadd2','hadd'),('calibration_hmul2','hmul')]:expected[(fam,(128,1,1),(32,1,1))]=(instr(op),4096*128,2*4096*128,2*4096*128)
casechecks=[];science=[];allaliases=[]
for case,spec in m['cases'].items():
 cp=p['cases'][case];need(cp['status']=='calibration_case_pass','case pass');need(cp['source_bindings_before']['bindings']==cp['source_tool_ELF_after']['bindings']==m['bindings']and cp['source_bindings_before']['all_unchanged']is True and cp['source_tool_ELF_after']['all_unchanged']is True,'case83beforeafter')
 ip,g=guard_payload(R/case/'identity.log',cp['identity_output_normalization'],R/case/'identity.payload.bin');idr=list(csv.reader(ip.decode().splitlines()));need(len(idr)==1 and len(idr[0])==2 and idr[0][0].strip()==constants['UUID']and idr[0][1].strip().lower()=='00000000:cb:00.0','raw identity UUID/PCI')
 if case=='precision':
  vp,vg=guard_payload(R/case/'version.log',cp['version_output_normalization'],R/case/'version.payload.bin');need(vp.decode()==p['actual_profiler_version_payload_text']and sha(vp)==p['actual_profiler_version_payload_sha256'],'version provenance');need(vp.decode().splitlines()==['NVIDIA (R) Nsight Compute Command Line Profiler','Copyright (c) 2018-2025 NVIDIA Corporation','Version 2025.2.0.0 (build 35613519) (public-release)'],'exact actual version')
 for label,log in [('profile','execution.log'),('identity_query','identity.log')]+([('profiler_version_process','version.log')]if case=='precision'else[]):
  rec=cp[label];need(rec['returncode']==0 and rec['stop_reason']is None and rec['external_gpu_activity']==[]and rec['own_descendants_cleaned']==[],'guarded process outcome '+label);need(rec['log_sha256']==sha(read(R/case/log)),'process log digest');argv=rec['executed_command'];split=argv.index('--');need(argv[:split]==['/usr/bin/python3',str(Path(m['source_root'])/'tools/v100_profile_write_guard.py'),'--allow-dir',str(R)],'executed guard argv');need(argv[split+1:]==cp['command']if label=='profile'else argv[split+1]==(m['ncu_executable']if label=='profiler_version_process'else m['nvidia_smi_executable']),'actual guarded executable/argv')
 command=cp['command'];need(command[0]==m['ncu_executable']and command[-1]==spec['binary'],'profile targettool/binary');need(command[command.index('--metrics')+1].split(',')==metrics,'requested command metrics')
 for k,v in [('--filter-mode','per-launch-config'),('--kernel-name-base','mangled'),('--kernel-id','::regex:.*:1'),('--clock-control','none'),('--cache-control','all'),('--replay-mode','kernel'),('--print-units','base')]:need(command[command.index(k)+1]==v,'profile selection/replay policy')
 log=read(R/case/'execution.log').decode();need([x for x in log.splitlines()if x.startswith('calibration_correct=')]==[spec['expected_host_check_line']]and cp['full_host_check']['line']==spec['expected_host_check_line']and cp['full_host_check']['pass']is True,'full actual hostcheck line');need(spec['expected_host_check_line']in read(spec['old_execution_log']).decode(),'old full hostcheck')
 new,newhash,warnings=table(R/case/'counters.csv');old,oldhash,_=table(spec['old_counters']);need(newhash==cp['counters_sha256']==cp['new_counter_sha256']and oldhash==cp['old_counter_sha256']==m['bindings'][spec['old_counters']],'oldnewcounter SHA')
 caseset={k:v for k,v in expected.items()if(case=='half_arithmetic')==(k[0]in ['calibration_hadd2','calibration_hmul2'])};need(set(new)==set(old)==set(caseset),'exact7plus2sourcegeometry')
 saved={ (r['source_family'],shape(r['identity']['Block Size']),shape(r['identity']['Grid Size'])):r for r in cp['counter_comparisons']};need(set(saved)==set(caseset),'savedscience identities')
 for key,(metric,count,physical,useful)in caseset.items():
  a=new[key];mm=a['numeric'];u=a['units'];need(all(mm.get(x)is not None and mm[x]>=0 for x in metrics),'full27 finite nonnegative');need(mm['gpu__time_duration.sum']>0,'positive duration');need(tuple(mm[f'device__attribute_pci_{x}_id']for x in ['domain','bus','device'])==(0,203,0),'rawPCI');need((mm['device__attribute_compute_capability_major'],mm['device__attribute_compute_capability_minor'])==(7,0),'raw sm70');need(mm[metric]==count,'source-known typed count')
  aliases=[]
  for metricname in metrics:
   unit=u[metricname]
   if metricname=='gpu__time_duration.sum':need(unit in ['nsecond','ns','usecond','us','msecond','ms','second','s'],'duration unit')
   elif metricname in freq:
    need(unit in ['hz','cycle/second']and mm[metricname]>0,'exact frequency dimensional unit');
    if unit=='hz':aliases.append({'metric':metricname,'raw_unit':'hz','validated_unit':'cycle/second','factor':1})
   elif metricname in ['dram__bytes_read.sum','dram__bytes_write.sum','dram__bytes.sum','lts__t_bytes.sum']:need(unit=='byte','byte unit')
   elif metricname in ['dram__bytes.sum.peak_sustained','lts__t_bytes.sum.peak_sustained']:need(unit=='byte/cycle','bytecycle unit')
   elif metricname.endswith('peak_sustained'):need(unit=='inst/cycle','instcycle unit')
   elif metricname.endswith('pct_of_peak_sustained_elapsed'):need(unit=='%','percent unit')
   else:need(unit=='inst','instruction unit')
  comparisons={}
  for name,value in old[key]['numeric'].items():
   if name.startswith('smsp__sass_thread_inst_executed_op_')or name in ['sm__inst_executed_pipe_tensor.sum','sm__inst_executed_pipe_tensor.sum.peak_sustained','sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained']:
    need(value is not None and mm.get(name)==value and u[name]==old[key]['units'][name],'old2022typed/arch invariant');comparisons[name]={'old2022':value,'new2025':mm[name]}
  s=saved[key];need(s['actual_count']==s['expected_count']==count and s['physical_executed_FLOPs']==physical and s['useful_source_FLOPs']==useful,'savedphysicaluseful');need(s['old2022_invariant_counter_comparisons']==comparisons,'full saved invariant map');need([{k:x[k]for k in ['metric','raw_unit','validated_unit','factor']}for x in s['validated_frequency_unit_aliases']]==aliases,'savedfrequency aliases')
  science.append({'source_family':key[0],'block':key[1],'grid':key[2],'metric':metric,'raw_instruction_count':count,'physical_FLOPs':physical,'useful_source_FLOPs':useful,'raw_duration':mm['gpu__time_duration.sum'],'duration_unit':u['gpu__time_duration.sum'],'requested_metric_count':len(metrics),'all_metrics_finite_units_valid':True,'old2022_invariant_count':len(comparisons),'frequency_aliases':aliases,'DRAM_read_plus_write_equals_direct_sum':mm['dram__bytes_read.sum']+mm['dram__bytes_write.sum']==mm['dram__bytes.sum']});allaliases+=aliases
 casechecks.append({'case':case,'rows':len(new),'host_check':spec['expected_host_check_line'],'new_counter_sha256':newhash,'old_counter_sha256':oldhash,'recorded83beforeafter':True,'identity_UUID_PCI_verified':True,'guard_schema_ABI':g['landlock_abi'],'guarded_profile_returncode':cp['profile']['returncode'],'foreign_activity_recorded':cp['profile']['external_gpu_activity'],'operational_seconds':cp['profile']['wall_seconds'],'raw_warnings_retained':warnings})
need(len(science)==9 and len(p['calibration_rows'])==9,'nine rows');need(p['calibration_rows']==p['cases']['precision']['counter_comparisons']+p['cases']['half_arithmetic']['counter_comparisons'],'globalscience exactjoins');need(p['physical_half_lane_factor']==2 and p['physical_half_fma_ops_per_instruction']==4 and p['tensor_FLOPs_per_HMMA884_instruction']==512,'conversion scope')
for path,raw in inputs.items():need(Path(path).read_bytes()==raw,'input changed during review '+path)
for x in bindings:need(hashfile(x['path'])==x['expected_sha256'],'83binding changed atfinish')
report={'status':'independent_saved003_scientific_evidence_consistent_pending_root_admission','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'calibration_provenance':{'path':str(R/'profile_provenance.json'),'sha256':sha(praw)},'manifest':{'path':p['manifest']['path'],'sha256':sha(mraw),'binding_count':83},'v5_constants':constants,'v5_candidate_sources':candidatefiles,'actual_version_payload':vp.decode(),'actual_version_payload_sha256':sha(vp),'cases':casechecks,'science_rows':science,'frequency_factor1_alias_instances':len(allaliases),'current83bindings':bindings,'all83_current_and_recorded_before_after_equal':True,'all_reviewed_input_bytes_stable':True,'no_builds_or_newGPU_processes':True,'no_live_method_or_registry_adoption':True,'limits':['Evidence validates exactexisting canary binaries on oneV100sm70 and recorded profiler build only; actual original compiler executable/version remains unknown.','Physical packed half2 factor2 lanes, HFMA4ops; scalar-half useful source work is smaller than physical count. Tensor512 is gated to exact Volta HMMA884 context.','Timing/clocks/traffic are not equality invariants across2022/2025; no empirical sustained ceiling, performance equivalence, fullROI or causal inference.','Recorded guard/policy/idle/foreign-monitor receipts and exactsource support operational compliance; no live GPU query or independent historical process replay performed.','V5 packer/helper itself was not run or imported; root must pack exact evidence and independently create approved note/receipt before live admission.'],'input_bindings':{p:{'sha256':sha(b),'bytes':len(b)}for p,b in inputs.items()}}
(O/'review.json').write_text(json.dumps(report,indent=2)+'\n');(O/'science_rows.json').write_text(json.dumps(science,indent=2)+'\n');print(json.dumps({'status':report['status'],'rows':len(science),'metric_count':len(metrics),'bindings':len(bindings),'aliases':len(allaliases),'review_sha256':sha((O/'review.json').read_bytes())}))
