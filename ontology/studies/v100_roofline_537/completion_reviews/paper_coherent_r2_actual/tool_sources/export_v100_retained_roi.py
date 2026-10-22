#!/usr/bin/env python3
"""CPU-only, fail-closed sums for explicitly source-counted trial2 kernel families.
Only triad/babelstream(dot)/jacobi named configurations are supported. No GPU,
build, source mutation, timing substitution or partial-sample extrapolation.
"""
import argparse,collections,csv,fcntl,gzip,hashlib,io,json,math,pathlib
csv.field_size_limit(64*1024*1024)
P=pathlib.Path
CASES={'triad':('triad_kernel_only_n4194304_r100_shmem0_v2',100),
       'babelstream':('fp32_n262144_dot_repeat10_v1',10),
       'jacobi':('float32_N2048_fixed20steps_with_norms_v1',20)}
TIME={'nsecond':1e-9,'ns':1e-9,'usecond':1e-6,'us':1e-6,'msecond':1e-3,'ms':1e-3,'second':1,'s':1}
def digest(x):return hashlib.sha256(x).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):P(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def counters(raw):
 lines=raw.decode().splitlines();start=next(i for i,l in enumerate(lines)if l.startswith('"ID"')or l.startswith('ID,'));rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))));groups={}
 if rows and 'Metric Name'not in rows[0]:
  units=next((r for r in rows if not r.get('ID')and r.get('gpu__time_duration.sum')in TIME),None)
  if units is None:raise ValueError('Wide CSV missing explicit time units')
  for r in rows:
   if not r.get('ID','').isdigit():continue
   key=tuple(r.get(k,'')for k in ['ID','Process ID','Context','Stream','Kernel Name'])
   if key in groups:raise ValueError('Duplicate wide launch identity')
   values={}
   for k,v in r.items():
    if '__'not in k:continue
    try:values[k]=float(v.replace(',',''))
    except ValueError:values[k]=None
   groups[key]={'identity':{k:r.get(k)for k in ['ID','Process ID','Context','Stream','Kernel Name','Device','Block Size','Grid Size']},'metrics':values,'units':{k:v for k,v in units.items()if '__'in k}}
  return list(groups.values())
 for r in rows:
  if not r.get('ID','').isdigit():continue
  key=tuple(r.get(k,'')for k in ['ID','Process ID','Context','Stream','Kernel Name'])
  g=groups.setdefault(key,{'identity':{k:r.get(k)for k in ['ID','Process ID','Context','Stream','Kernel Name','Device','Block Size','Grid Size']},'metrics':{},'units':{}});n=r['Metric Name']
  if n in g['metrics']:raise ValueError('Duplicate kernel metric')
  try:v=float(r['Metric Value'].replace(',',''))
  except ValueError:v=None
  g['metrics'][n]=v;g['units'][n]=r['Metric Unit']
 return list(groups.values())
def metric(g,name,unit):
 v=g['metrics'].get(name)
 if not isinstance(v,(int,float))or not math.isfinite(v)or v<0 or g['units'].get(name)!=unit:raise ValueError('Missing/invalid counter '+name)
 return v
PROTOCOL='''# Source-counted retained-trial kernel aggregation (version 1)

## Scope and acceptance

This exporter accepts only the declared triad N4194304/R100, BabelStream
N262144/dot/R10, and Jacobi N2048/20-step configurations. It uses the current
selected capture, never substitutes a failed latest attempt, and never scales
one sampled kernel by a source repeat count. A complete family requires an
unchanged source/config/result/selector recipe, successful whole profiler
process and original dual-reference numerical receipt, five fresh positive CPU
self-check rows, a source ordinal window for retained trial2, the exact expected
number of distinct raw kernel identities, and explicit per-kernel timed-stage
membership annotations. Source-ordinal proof is not a captured whole-run trace.
Preserved raw counter/manifest/selector/source hashes make each admission auditable.

## Allowed totals

For all actual selected scientific launches i, retain precision separately and
compute F_p = sum_i F_(i,p), B_DRAM = sum_i(read_i+write_i), B_L2 = sum_i L2_i,
and T_device_replay = sum_i duration_i. Scalar FP32/64 work is add+mul+2*FMA;
FP16/tensor work may not be inferred without calibrated arithmetic conventions.
This limited exporter rejects nonzero half/double/tensor arithmetic rather than
silently adding unlike peaks. Conventional scalar counts exclude atomic/SFU,
integer/address/control work. Kernel multiplicity is counted once per unique
raw identity; profiler hardware replay passes are not extra scientific launches.

The integrated work/traffic coordinate is I_p = F_p/B, **not the mean of kernel
intensities**. Device replay rate F_p/T_device_replay is labeled as such; summing
individual launch rates or equal-weight averaging rates is invalid. The source
runs these cases serially. If future stages overlap, summed duration is device
service time, not union busy time, makespan, or elapsed time. Raw cold-replay
traffic is not the uninstrumented warm-cache trajectory traffic. Complete
source-counted kernel work under this replay protocol can be shown independently
of any claim about full elapsed ROI behavior.

## Timing and ceilings

Original five checked trials remain unchanged: mean trials2..5 is reported per
internal repeat/step. To recover the declared host timer interval, multiply
reported seconds by source repeats (100/10/20). This host interval includes
launch/control/synchronization; Jacobi also includes scalar D2H, host sqrt and
error reset. Fresh profiler trial2 host clocks include collection/replay overhead.
Neither original nor instrumented host time may be used as the denominator for
these new replay counters. Show original means, fresh clock interval and summed
kernel replay duration as separate quantities. Their difference is not a pure
host overhead estimate under profiling.

Keep per-launch clock-adjusted modeled ceilings/ridges as ranges and per-launch
geometric regimes. Do not average ceiling FLOP/s, ridge, intensity or occupancy
into a benchmark diagnosis. No aggregate geometric regime or causal bottleneck
is asserted here: causal bottleneck remains unknown. For a future fully mapped
serial homogeneous stage, a clearly named model may form sum_i max(F_i/P_i,
B_i/BW_i) as a per-launch ideal-service lower bound; that is not an observed
elapsed-time roof or a measured sustained device peak. Mixed precisions require
separate work vectors/projections, not summed peaks.

## Completeness boundaries

Triad's named study contains only 100 serial scientific triad launches within the
host timer; source stream/tile/transfer benchmark is excluded. A complete family
therefore covers the named scientific device work, not the original full CLI or
elapsed-host ROI. BabelStream here means dot only: not six STREAM operations or
host final scalar reduction. N262144 working set fits L2; cold NCU DRAM traffic
must not be called normal STREAM bandwidth. Jacobi's 20 stencil calls may cover
all selected stencil computation, but timer-included resets/memsets, transfers
and host norms remain outside that counter family. Atomic accumulation also has
cost/work not represented by scalar add/mul/FMA totals. Neither complete stencil
family nor a memory-side projection proves full-solver/causal bottleneck.

Original accepted numerical outputs and dual-reference receipts are separate
from new profiler self-check logs. Disposable new outputs were manifest-recorded
then removed by the collector; no new direct-artifact correctness certification
is inferred. Old rejected/partial captures remain unchanged and unextrapolated.
'''
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1]);a.add_argument('--output-root',type=P,required=True);a.add_argument('--offline-snapshot',action='store_true',help='Require quiescent files; rehash all inputs afterward instead of shared metadata lock.');z=a.parse_args();root=z.source_root.resolve();out=z.output_root.absolute()
 if not out.resolve().is_relative_to(P('/tmp'))or out==P('/tmp')or out.exists()or out.is_symlink():a.error('Fresh /tmp output required')
 lock=None
 if not z.offline_snapshot:
  lock=P('/tmp/hecbench-roofline-metadata.lock').open('rb');fcntl.flock(lock,fcntl.LOCK_SH)
 inputs={};rows=[]
 def bound(p,expected=None):
  p=P(p);p=p if p.is_absolute()else root/p;h=sha(p)
  if expected and h!=expected:raise ValueError('Hash mismatch '+str(p))
  inputs[str(p)]=h;return p
 def read(p,expected=None):return bound(p,expected).read_bytes()
 def load(p,expected=None):return json.loads(read(p,expected))
 try:
  study=root/'ontology/studies/v100_roofline_537'
  for b,(config,count)in CASES.items():
   ledger=load(study/'benchmarks'/f'{b}.json')
   for m in ['cuda','cuda_jl','ka','jacc']:
    state=ledger['models'][m];row={'benchmark':b,'model':m,'configuration_id':config,'selected_profile':state.get('profile_ids'),'latest_profile_attempt':state.get('latest_profile_attempt'),'selected_roi_coverage':state.get('roi_coverage'),'selected_sampled_kernel_count':state.get('sampled_kernel_count'),'expected_scientific_launches':count,'accepted_complete_scientific_kernel_family':False,'complete_elapsed_host_ROI':False,'causal_bottleneck':'unknown','aggregate_geometric_regime':None,'missingness_reasons':[]};rows.append(row)
    try:
     manifestpath=state.get('profile_evidence_manifest')
     if not manifestpath:raise ValueError('No selected evidence manifest')
     manifest=load(manifestpath)
     frozen={}
     for f in manifest['files']:
      compressed=read(f['path'],f['compressed_sha256']);raw=gzip.decompress(compressed)
      if digest(raw)!=f['uncompressed_sha256']:raise ValueError('Uncompressed archived hash mismatch')
      frozen[f['original_name']]=raw
     prov=json.loads(frozen['profile_provenance.json']);plan=json.loads(frozen['plan.json']);row['capture_provenance']={'manifest':str(root/manifestpath),'manifest_sha256':inputs[str(root/manifestpath)],'profile_provenance_sha256':digest(frozen['profile_provenance.json']),'counter_sha256':digest(frozen.get('counters.csv',b'')),'plan_sha256':digest(frozen['plan.json']),'collector_sha256':prov.get('collector_sha256'),'parser_sha256':prov.get('parser_sha256'),'device_uuid':prov.get('gpu_uuid'),'source_binding_recheck':prov.get('binding_recheck_after')}
     if prov.get('status')!='profile_process_completed'or prov.get('profile',{}).get('returncode')!=0:raise ValueError('Selected process did not complete')
     if prov.get('binding_recheck_after')!='unchanged':raise ValueError('Source bindings not unchanged')
     if not prov.get('profile_device_identity_verified'):raise ValueError('No verified device identity')
     if prov.get('counters_sha256')!=digest(frozen['counters.csv']):raise ValueError('Raw counter receipt mismatch')
     if digest(frozen['plan.json'])!=prov['plan_sha256']:raise ValueError('Plan receipt mismatch')
     if plan['configuration_id']!=config:raise ValueError('Not the explicitly reviewed named configuration')
     original=load(plan['reference_result'],plan['reference_result_sha256']);rawconfig=P(plan['reference_result']).with_name('config.json');bound(rawconfig,plan['raw_config_sha256'])
     row['original_numerical_receipt']={'result':plan['reference_result'],'result_sha256':plan['reference_result_sha256'],'config':str(rawconfig),'config_sha256':plan['raw_config_sha256'],'reference_agreement':original.get('reference_agreement'),'variant_verified':original.get('variant_verified'),'primary_mean_seconds_per_internal_repeat':original['models'][m].get('mean_seconds'),'raw_trial2_seconds_per_internal_repeat':original['models'][m]['trials'][1].get('time_seconds'),'original_saved_artifacts_rehashed_here':False}
     if original.get('reference_agreement')is not True or not all(original.get('variant_verified',{}).get(v)is True for v in ['ka','jacc']):raise ValueError('Original dual reference numerical gate missing')
     oldtrials=original['models'][m]['trials']
     if len(oldtrials)!=5 or any(t.get('correct')is not True for t in oldtrials):raise ValueError('Original five independent checks incomplete')
     row['source_current_bindings']={}
     for p,h in {**plan['execution_bindings'],**prov.get('additional_current_profile_input_bindings',{})}.items():bound(p,h);row['source_current_bindings'][p]=h
     sel=prov.get('selection_override')or{}
     if sel.get('retained_trial')!=2 or prov.get('retained_window_expected_count_confirmed')is not True or prov.get('selection_expected_launch_count_matches')is not True:raise ValueError('No complete source-counted retained-trial2 capture')
     selection=load(sel['path'],sel['sha256']);recipe=load(sel['source_recipe']['path'],sel['source_recipe']['sha256']);bound(selection['fixture_description']['path'],selection['fixture_description']['sha256'])
     source=next(x for x in selection['source_records']if x['model']==m)
     if source['configuration_id']!=config or source['reference_result']['sha256']!=plan['reference_result_sha256']or source['raw_config']['sha256']!=plan['raw_config_sha256']:raise ValueError('Source ordinal recipe contract mismatch')
     for x in source['execution_bindings']+source.get('current_supplemental_sources',[]):bound(x['path'],x['sha256'])
     chosen=selection['selection'][m];ledgerproof=source['launch_ledger']
     if chosen.count('--launch-skip')!=1 or chosen.count('--launch-count')!=1 or chosen.count('--filter-mode')!=1:raise ValueError('Ambiguous source window selection')
     if b=='jacobi':
      start=ledgerproof['trial2_scientific_start']
      if ledgerproof['scientific_per_trial']!=20 or ledgerproof['scientific_warmup']!=1 or start!=22:raise ValueError('Jacobi source ordinal schema mismatch')
     else:start=ledgerproof['trial2_start']
     if chosen[chosen.index('--filter-mode')+1]!='global' or int(chosen[chosen.index('--launch-count')+1])!=count or int(chosen[chosen.index('--launch-skip')+1])!=start-1:raise ValueError('Source ordinal selection mismatch')
     if '--kill'not in chosen or chosen[chosen.index('--kill')+1]not in ['0','no']:raise ValueError('Application completion not retained')
     command=prov['command']
     if not any(command[i:i+len(chosen)]==chosen for i in range(len(command)-len(chosen)+1)):raise ValueError('Executed selector differs from source window')
     row['source_launch_proof']={'selector_path':sel['path'],'selector_sha256':sel['sha256'],'recipe':sel['source_recipe'],'source_ROI':source['source_ROI'],'launch_ledger':source['launch_ledger'],'selection':selection['selection'][m],'proof_kind':'source_launch_ordinal_not_dynamic_range_trace'}
     if selection['configuration_id']!=config or selection['expected_matching_launches']!=count or sel['expected_matching_launches']!=count:raise ValueError('Expected multiplicity mismatch')
     fresh=[]
     for line in frozen['execution.log'].decode().splitlines():
      if line.startswith('STUDY_TRIAL '):fresh.append(json.loads(line[len('STUDY_TRIAL '):]))
     if [x.get('trial')for x in fresh]!=[1,2,3,4,5]or any(x.get('correct')is not True or x.get('checked_values',0)<=0 for x in fresh):raise ValueError('Five complete fresh CPU self-check rows absent')
     repeats=100 if b=='triad'else 10 if b=='babelstream'else 20
     row['timing_normalization']={'reported_seconds':'per internal repeat/step','multiply_by_to_recover_source_timer_interval':repeats,'original_trial2_host_interval_seconds':oldtrials[1]['time_seconds']*repeats,'primary_last4_mean_host_interval_seconds':original['models'][m]['mean_seconds']*repeats,'source_timer_is_inclusive_of_launch_and_completion':True,'profile_counter_rate_denominator_is_only_sum_replayed_kernel_durations':True}
     if any(x.get('internal_repeats')!=repeats for x in fresh):raise ValueError('Timing normalization mismatch')
     row['fresh_numerical_qualification']={'five_inprocess_CPU_selfchecks_pass':True,'new_direct_dual_artifact_comparison':False,'output_retention':'collector auxiliary manifest only; original accepted numerical artifacts remain separately authoritative','trial2_instrumented_host_seconds_per_repeat':fresh[1]['time_seconds'],'trial2_instrumented_host_interval_seconds':fresh[1]['time_seconds']*repeats,'not_a_performance_replacement':True}
     linked=state.get('kernel_observations')or{};annotatedraw=gzip.decompress(read(linked['path'],linked['sha256']))
     if digest(annotatedraw)!=linked['uncompressed_sha256']:raise ValueError('Annotated hash mismatch')
     ann=json.loads(annotatedraw)['records'];raw=counters(frozen['counters.csv'])
     if len(ann)!=count or len(raw)!=count or prov.get('kernel_count')!=count:raise ValueError('Actual scientific launch multiplicity mismatch')
     ak=lambda x:tuple(str(x.get(k,''))for k in ['ID','Process ID','Context','Stream','Kernel Name'])
     amap={ak(x['identity']):x for x in ann}
     if len(amap)!=count:raise ValueError('Duplicate annotated launch identity')
     F=0.;B=0.;R=0.;W=0.;L2=0.;T=0.;regimes=collections.Counter();ridge=[];records=[]
     for g in raw:
      arow=amap[ak(g['identity'])];ctx=arow['scope_context']
      if not all(ctx.get(x)is True for x in ['source_context_bound_before_after','sample_in_primary_timed_invocation','semantic_family_in_primary_ROI'])or ctx.get('retained_trial')!=2:raise ValueError('Unassigned or non-timed source stage')
      if ctx.get('reference_result_sha256')!=plan['reference_result_sha256']or ctx.get('configuration_id')!=config:raise ValueError('Kernel context contract mismatch')
      for op in ['hadd','hmul','hfma','dadd','dmul','dfma']:
       if metric(g,'smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum','inst')!=0:raise ValueError('Mixed/non-FP32 vector outside this strict exporter')
      if metric(g,'sm__inst_executed_pipe_tensor.sum','inst')!=0:raise ValueError('Tensor work outside this strict exporter')
      f=sum(w*metric(g,'smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum','inst')for op,w in [('fadd',1),('fmul',1),('ffma',2)])
      r=metric(g,'dram__bytes_read.sum','byte');w=metric(g,'dram__bytes_write.sum','byte');direct=metric(g,'dram__bytes.sum','byte')
      if direct!=r+w:raise ValueError('Direct DRAM total mismatch')
      t=metric(g,'gpu__time_duration.sum',g['units'].get('gpu__time_duration.sum'));factor=TIME.get(g['units'].get('gpu__time_duration.sum'))
      if not factor or t<=0:raise ValueError('Unknown/nonpositive duration')
      t*=factor;l2=metric(g,'lts__t_bytes.sum','byte');proj=arow['scalar_precision_projections']['fp32']
      if proj['executed_scalar_flops']!=f or arow['dram_bytes']!=direct or not math.isclose(arow['duration_seconds'],t,rel_tol=1e-12):raise ValueError('Derived/raw totals disagree')
      F+=f;R+=r;W+=w;B+=direct;L2+=l2;T+=t;regimes[proj['roofline_regime']]+=1
      if proj.get('ridge_flop_per_byte')is not None:ridge.append(proj['ridge_flop_per_byte'])
      records.append({'identity':g['identity'],'fp32_executed_flops':f,'dram_read_bytes':r,'dram_write_bytes':w,'L2_transaction_bytes':l2,'duration_seconds':t,'geometric_regime':proj['roofline_regime']})
     row.update(accepted_complete_scientific_kernel_family=True,work_traffic_scope='all source-counted selected scientific launches under cold replay, not complete elapsed-host ROI',totals={'actual_scientific_launches':count,'fp32_executed_flops':F,'dram_read_bytes':R,'dram_write_bytes':W,'dram_bytes':B,'L2_transaction_bytes':L2,'sum_device_kernel_replay_seconds':T,'fp32_flop_per_DRAM_byte':F/B if B else None,'fp32_flop_per_L2_byte':F/L2 if L2 else None,'fp32_flops_per_summed_kernel_replay_second':F/T,'aggregate_ridge':None,'aggregate_regime':None},per_launch_geometric_regime_counts=dict(regimes),per_launch_ridge_range=[min(ridge),max(ridge)]if ridge else None,scientific_kernel_records=records)
     if b=='triad':row['separate_source_useful_work_check']={'definition':'2*n*repeats, one add and one multiply per element','expected_flops':2*4194304*100,'matches_executed_fp32_flops':F==2*4194304*100}
     row['whole_named_scientific_device_stage_covered']=b in ['triad','babelstream'];row['whole_elapsed_timer_work_traffic_covered']=False
    except (ValueError,KeyError,StopIteration,FileNotFoundError)as e:row['missingness_reasons'].append(str(e))
  bound(P(__file__).resolve())
  for p,h in inputs.items():
   if sha(p)!=h:raise ValueError('Input changed during export: '+p)
 finally:
  if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 out.mkdir(parents=True);dump(out/'audit.json',{'schema_version':1,'scope':'three named trial2 scientific kernel-family audits only','input_hashes':inputs,'records':rows,'eligible_family_count':sum(x['accepted_complete_scientific_kernel_family']for x in rows),'elapsed_host_ROI_positions':0,'whole_benchmark_causal_claims':0,'original_primary_timings_changed':False,'shared_metadata_changed':False,'snapshot_method':'offline_all_input_rehash'if z.offline_snapshot else 'shared_metadata_lock_and_all_input_rehash'})
 (out/'PROTOCOL.md').write_text(PROTOCOL)
 with(out/'summary.csv').open('w')as f:
  w=csv.writer(f);w.writerow(['benchmark','model','complete_scientific_family','launches','fp32_flops','DRAM_bytes','L2_bytes','sum_kernel_replay_seconds','FP32_FLOP_per_DRAM_byte','missingness'])
  for r in rows:
   t=r.get('totals',{});w.writerow([r['benchmark'],r['model'],r['accepted_complete_scientific_kernel_family'],t.get('actual_scientific_launches'),t.get('fp32_executed_flops'),t.get('dram_bytes'),t.get('L2_transaction_bytes'),t.get('sum_device_kernel_replay_seconds'),t.get('fp32_flop_per_DRAM_byte'),'; '.join(r['missingness_reasons'])])
 print(json.dumps({'audit':str(out/'audit.json'),'eligible_scientific_families':sum(x['accepted_complete_scientific_kernel_family']for x in rows),'elapsed_ROI_positions':0}))
if __name__=='__main__':main()
