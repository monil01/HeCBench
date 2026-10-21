"""Independent CPU diagnostic of saved002 only; imports no study/proposed module."""
from pathlib import Path
import csv,io,json,hashlib,math,re
O=Path(__file__).parent;RAW=Path('/tmp/hecbench-roofline-v100-20261020');D=RAW/'method_calibration_ncu2025_2_0_002';PRE=O.parent/'profiler2025_calibration002';inputs={}
def read(p):
 p=Path(p);b=p.read_bytes();inputs[str(p)]=b;return b
def sha(b):return hashlib.sha256(b).hexdigest()
def need(x,s):
 if not x:raise ValueError(s)
def table(p):
 raw=read(p);ls=raw.decode().splitlines();i=next(i for i,l in enumerate(ls)if l.startswith('"ID"')or l.startswith('ID,'));reader=csv.DictReader(io.StringIO('\n'.join(ls[i:])));rows=list(reader);need('Metric Name'not in reader.fieldnames,'Independent audit expected actual wide format');unitrow=next(r for r in rows if not r['ID']and r.get('gpu__time_duration.sum'));records=[r for r in rows if r.get('ID','').isdigit()];return reader.fieldnames,unitrow,records
p=json.loads(read(D/'profile_provenance.json'));need(p['status']=='new_profiler_calibration_failed_no_adoption','Not actual rejected002');need(p['error']=="RuntimeError('Metric unit mismatch dram__cycles_elapsed.avg.per_second hz expected cycle/second')",'Unexpected failure')
man=json.loads(read(PRE/'source_binding_manifest.json'));cp=p['cases']['precision'];need('half_arithmetic'not in p['cases'],'Unexpected second case');need(cp['profile']['returncode']==0 and cp['profile']['stop_reason']is None,'Precision underlying process failed')
log=read(D/'precision/execution.log');need(sha(log)==cp['profile']['log_sha256'],'Full executionlog hash mismatch');need([l for l in log.decode().splitlines()if l.startswith('calibration_correct=')]==[man['cases']['precision']['expected_host_check_line']],'Full host values not checked')
newhead,nu,nr=table(D/'precision/counters.csv');oldhead,ou,oldrows=table(RAW/'precision_calibration/counters.csv');need(sha(inputs[str(D/'precision/counters.csv')])==cp['counters_sha256'],'Actual002counter hash mismatch')
metrics=p['requested_metrics'];need(len(metrics)==len(set(metrics))==27,'Full27inventory changed');frequency={'dram__cycles_elapsed.avg.per_second','sm__cycles_elapsed.avg.per_second','lts__cycles_elapsed.avg.per_second'};units=[]
for m in metrics:
 actual=nu[m];expect=('inst'if m.startswith(('smsp__sass_thread_inst_executed_op_','sm__inst_executed_pipe_tensor.sum'))and not m.endswith('peak_sustained')else 'byte'if m in ['dram__bytes_read.sum','dram__bytes_write.sum','dram__bytes.sum','lts__t_bytes.sum']else 'cycle/second'if m.endswith('avg.per_second')else '%'if m.endswith('pct_of_peak_sustained_elapsed')else 'byte/cycle'if m in ['dram__bytes.sum.peak_sustained','lts__t_bytes.sum.peak_sustained']else 'inst/cycle'if m.endswith('peak_sustained')else None)
 okay=(actual in ['nsecond','ns','usecond','us','msecond','ms','second','s'])if m=='gpu__time_duration.sum'else(actual in ['cycle/second','hz'])if m in frequency else actual==expect
 need(okay,'Other unsupported unit drift '+m+' '+actual)
 units.append({'metric':m,'actual002_unit':actual,'old2022_unit':ou.get(m)if m in oldhead else None,'old2022_metric_captured':m in oldhead,'expected_dimension':expect if m!='gpu__time_duration.sum'else'time','eligible_frequency_alias':m in frequency and actual=='hz','numeric_factor':1 if m in frequency and actual=='hz'else None,'policy':'exact_three_name_hz_alias'if m in frequency and actual=='hz'else'already_supported_time_spelling'if m=='gpu__time_duration.sum'and actual!=ou.get(m)else'no_new_alias'})
def key(r):
 name=r['Kernel Name'];f=next((s for s in ['calibration_half2','calibration_half','calibration_tensor']if re.search(r'\b'+s+r'\(',name)),None)
 if 'selector_typed<float>'in name:f='selector_float'
 if 'selector_typed<double>'in name:f='selector_double'
 shape=lambda s:tuple(map(int,re.findall(r'\d+',s)))
 return f,shape(r['Block Size']),shape(r['Grid Size'])
def num(v):
 x=float(v.replace(',',''));need(math.isfinite(x)and x>=0,'Nonfinite/negative metric');return x
oldmap={key(r):r for r in oldrows};newmap={key(r):r for r in nr}
expected={('calibration_half2',(128,1,1),(32,1,1)):('hfma',524288),('calibration_half',(128,1,1),(32,1,1)):('hfma',524288),('calibration_tensor',(32,1,1),(1,1,1)):('tensor',2048),('selector_float',(128,1,1),(32,1,1)):('ffma',4096),('selector_double',(128,1,1),(32,1,1)):('dfma',4096),('selector_float',(256,1,1),(16,1,1)):('ffma',4096),('selector_double',(256,1,1),(16,1,1)):('dfma',4096)}
need(len(nr)==len(oldrows)==7 and set(newmap)==set(oldmap)==set(expected),'Exact7geometries missing');details=[]
for k,r in newmap.items():
 values={m:num(r[m])for m in metrics};need(values['gpu__time_duration.sum']>0,'No positive time');need(tuple(num(r['device__attribute_pci_'+n+'_id'])for n in ['domain','bus','device'])==(0,203,0),'ActualPCI mismatch');need((num(r['device__attribute_compute_capability_major']),num(r['device__attribute_compute_capability_minor']))==(7,0),'Not sm70')
 op,want=expected[k];metric='sm__inst_executed_pipe_tensor.sum'if op=='tensor'else'smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum';need(values[metric]==want,'Sourcecount mismatch');old=oldmap[k];compared=[]
 for m in oldhead:
  if m.startswith('smsp__sass_thread_inst_executed_op_')or m in ['sm__inst_executed_pipe_tensor.sum','sm__inst_executed_pipe_tensor.sum.peak_sustained','sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained']:
   need(m in values and num(old[m])==values[m]and ou[m]==nu[m],'Old/new invariant mismatch '+m);compared.append(m)
 details.append({'family':k[0],'block':k[1],'grid':k[2],'source_known_metric':metric,'actual_count':values[metric],'expected_count':want,'all27_finite_present':True,'old_invariant_metrics_compared':compared,'actualPCI':[0,203,0],'CC':[7,0]})
section=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Compute/sections/SpeedOfLight_HierarchicalSingleRooflineChart.section');source=read(section).decode();slines=source.splitlines();evidence=[]
for i,l in enumerate(slines):
 if any('Name: "'+m+'"'in l for m in frequency)and i>0 and 'Frequency'in slines[i-1]:evidence.append({'line':i+1,'preceding_label':slines[i-1],'metric_line':l})
need({m for m in frequency if any(m in e['metric_line']for e in evidence)}==frequency,'InstalledSDK frequency labels missing')
# Preserve small original002 and input-source references privately; no old mutations.
archive=O/'preserved002';archive.mkdir(exist_ok=False)
for src,name in [(D/'profile_provenance.json','profile_provenance.json'),(D/'precision/counters.csv','precision_counters.csv'),(D/'precision/execution.log','precision_execution.log'),(PRE/'calibrate_controller.py','controller_source.py'),(PRE/'source_binding_manifest.json','source_binding_manifest.json')]:
 b=read(src);(archive/name).write_bytes(b)
for path,b in inputs.items():need(Path(path).read_bytes()==b,'Review input changed')
result={'status':'diagnostic_saved002_precision7row_reinterpretation_passed_NOT_9row_calibration_or_adoption','original002_status':p['status'],'original002_error':p['error'],'other_case_executed':False,'calibration_rows_present':7,'required_complete_calibration_rows':9,'full_original_precision_host_values_passed':True,'all27_metric_unit_review':units,'new_frequency_aliases':sorted(frequency),'factor':1,'raw_units_or_values_changed':False,'row_diagnostics':details,'installedSDK_source_evidence':{'path':str(section),'sha256':sha(inputs[str(section)]),'frequency_labels':evidence},'official_sources':[{'url':'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html','claim':'Per-second suffix denotes operations per second; current official general metric structure, not substitute for installed2025bytes.'},{'url':'https://forums.developer.nvidia.com/t/what-is-gpc-cycles-elapsed-max/198128','claim':'NVIDIA response explains elapsed-cycle per_second metrics are clock frequency.'},{'url':'https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html','retrieval':'Web tool redirected to inaccessible archived domain; no archived text claimed read.'}],'limits':'CPU diagnostic checks actual saved7precision rows only under a prospectively stated narrow dimensional alias. It does not rerun targets, execute proposed modules, amend original rejected002criterion/status, or stand in for fresh003all9pass. Old unavailable metrics remain null/uncaptured, never zero. Frequency aliases do not normalize other metrics, prefixes or scaled units.','input_sha256':{p:sha(b)for p,b in inputs.items()},'newGPU_build_tests_imports':0}
(O/'actual002_unit_review.json').write_text(json.dumps(result,indent=2)+'\n')
with (O/'metric_unit_pairs.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(units[0]));w.writeheader();w.writerows(units)
print(json.dumps({'actual_rows_diagnosed':len(details),'units_reviewed':len(units),'unsupported_other_unit_drift':0,'actual002_unit_review_sha256':sha((O/'actual002_unit_review.json').read_bytes()),'no_newGPU_or_complete_calibration':True}))
