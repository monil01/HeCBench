#!/usr/bin/env python3
"""Parse NCU2022 raw CSV into auditable per-kernel precision projections (version 1).
No profiling, source writes, aggregate benchmark diagnoses, or external dependencies.
"""
import argparse,csv,hashlib,io,json,math,pathlib
KINDS={'fp16':('hadd','hmul','hfma'),'fp32':('fadd','fmul','ffma'),'fp64':('dadd','dmul','dfma')}
TIME={'nsecond':1e-9,'ns':1e-9,'usecond':1e-6,'us':1e-6,'msecond':1e-3,'ms':1e-3,'second':1,'s':1}
def number(x):
 try:v=float(str(x).strip().replace(',',''))
 except (TypeError,ValueError):return None
 return v if math.isfinite(v) and v>=0 else None

def read_csv(path):
 raw=path.read_bytes();ls=raw.decode(errors='replace').splitlines()
 start=next((i for i,l in enumerate(ls) if l.startswith('"ID"') or l.startswith('ID,')),None)
 if start is None:raise ValueError('No NCU CSV ID header')
 rows=list(csv.DictReader(io.StringIO('\n'.join(ls[start:]))));groups={}
 if rows and 'Metric Name' in rows[0]:
  for r in rows:
   if not r.get('ID','').isdigit():continue
   key=tuple(r.get(k,'')for k in ['ID','Process ID','Context','Stream','Kernel Name'])
   g=groups.setdefault(key,{'identity':{k:r.get(k)for k in ['ID','Process ID','Kernel Name','Context','Stream','Device','Block Size','Grid Size']},'metrics':{},'units':{}})
   name=r['Metric Name']
   if name in g['metrics']:raise ValueError('Duplicate metric in kernel row: '+name)
   g['metrics'][name]=number(r.get('Metric Value'));g['units'][name]=r.get('Metric Unit','')
 else:
  unitrow=next((r for r in rows if not r.get('ID') and r.get('gpu__time_duration.sum')in TIME),None)
  if unitrow is None:raise ValueError('Wide CSV missing explicit recognized time unit row')
  for r in rows:
   if not r.get('ID','').isdigit():continue
   key=tuple(r.get(k,'')for k in ['ID','Process ID','Context','Stream','Kernel Name'])
   if key in groups:raise ValueError('Duplicate wide kernel identity; no rows[-1] fallback')
   groups[key]={'identity':{k:r.get(k)for k in ['ID','Process ID','Kernel Name','Context','Stream','Device','Block Size','Grid Size']},'metrics':{k:number(v)for k,v in r.items()if '__'in k},'units':{k:v for k,v in unitrow.items()if '__'in k}}
 return list(groups.values()),hashlib.sha256(raw).hexdigest()

def derive(g,half_lanes=None,half_provenance=None,context=None,tensor_flops_per_instruction=None,tensor_convention_evidence=None):
 m=g['metrics'];u=g.get('units',{});warnings=[];ctx=context or {}
 sm70=ctx.get('gpu_sm70') is True or ctx.get('compute_capability') in ['7.0','sm_70'] or ctx.get('gpu_architecture')=='sm_70'
 def val(k):return number(m.get(k))
 def metric(k,unit):
  x=val(k)
  if x is not None and u.get(k)!=unit:
   warnings.append('Unit unavailable/mismatch: '+k+' expected '+unit+' got '+str(u.get(k)));return None
  return x
 t=val('gpu__time_duration.sum');tf=TIME.get(u.get('gpu__time_duration.sum'))
 dt=t*tf if t is not None and tf is not None and t>0 else None
 if dt is None:warnings.append('Missing/nonpositive duration or unknown duration unit')
 ops={};raw={};peaks={}
 clocks=metric('sm__cycles_elapsed.avg.per_second','cycle/second')
 for precision,names in KINDS.items():
  counts=[metric('smsp__sass_thread_inst_executed_op_'+n+'_pred_on.sum','inst')for n in names]
  raw[precision]=dict(zip(names,counts));base=sum((1,1,2)[i]*v for i,v in enumerate(counts))if all(v is not None for v in counts)else None
  # Half lane convention is deliberately gated. Presence remains detectable without a FLOP conversion.
  lanes=half_lanes if precision=='fp16' else 1
  valid_half=precision!='fp16' or (lanes in (1,2) and bool(half_provenance))
  ops[precision]=base*lanes if base is not None and valid_half else (0 if base==0 else None)
  p=metric('sm__sass_thread_inst_executed_op_'+names[2]+'_pred_on.sum.peak_sustained','inst/cycle')
  peaks[precision]=2*lanes*p*clocks if valid_half and lanes is not None and p is not None and clocks is not None else None
 tensor=metric('sm__inst_executed_pipe_tensor.sum','inst')
 detected={p:(any(v>0 for v in raw[p].values() if v is not None)if all(v is not None for v in raw[p].values())else None)for p in KINDS}
 detected['tensor']=tensor>0 if tensor is not None else None
 positive=[p for p,v in detected.items()if v]
 allknown=all(v is not None for v in detected.values())
 precision_class=('mixed_'+ '_'.join(positive) if len(positive)>1 else positive[0] if positive else 'no_counted_FP')if allknown else 'incomplete_precision_vector'
 read=metric('dram__bytes_read.sum','byte');write=metric('dram__bytes_write.sum','byte');dram=read+write if read is not None and write is not None else metric('dram__bytes.sum','byte')
 direct=metric('dram__bytes.sum','byte')
 if direct is not None and read is not None and write is not None and direct!=read+write:warnings.append('DRAM sum vs read+write mismatch; retained read+write, investigate')
 bcycle=metric('dram__bytes.sum.peak_sustained','byte/cycle');bclock=metric('dram__cycles_elapsed.avg.per_second','cycle/second');bw=bcycle*bclock if bcycle is not None and bclock is not None else None
 l2=metric('lts__t_bytes.sum','byte');l2cycle=metric('lts__t_bytes.sum.peak_sustained','byte/cycle');l2clock=metric('lts__cycles_elapsed.avg.per_second','cycle/second')
 l2bw=l2cycle*l2clock if l2cycle is not None and l2clock is not None else None
 tensor_valid=tensor_flops_per_instruction==512 and bool(tensor_convention_evidence) and sm70
 tensor_flops=tensor*tensor_flops_per_instruction if tensor is not None and tensor_valid else (0 if tensor==0 else None)
 tensor_peak_inst=metric('sm__inst_executed_pipe_tensor.sum.peak_sustained','inst/cycle')
 tensor_peak=tensor_peak_inst*tensor_flops_per_instruction*clocks if tensor_valid and tensor_peak_inst is not None and clocks is not None else None
 def project(n,C,traffic,bandwidth,tier):
  I=n/traffic if n is not None and n>0 and traffic is not None and traffic>0 else None
  rate=n/dt if n is not None and n>0 and dt else None
  ridge=C/bandwidth if C is not None and C>0 and bandwidth is not None and bandwidth>0 else None
  ratio=I/ridge if I is not None and ridge else None
  regime='not_applicable' if n==0 else 'unclassified_missing_counter_or_ceiling' if ratio is None else 'near_ridge' if .8<=ratio<=1.25 else 'memory_side' if ratio<.8 else 'compute_side'
  roof=min(C,bandwidth*I) if C is not None and bandwidth is not None and I is not None else None
  if traffic==0 and n is not None and n>0:regime='undefined_'+tier+'_intensity_zero_traffic'
  return {'executed_flops':n,'memory_tier':tier,'measured_bytes':traffic,'arithmetic_intensity_flop_per_byte':I,'achieved_flops_per_second':rate,'modeled_clock_adjusted_peak_flops_per_second':C,'modeled_clock_adjusted_peak_bandwidth_bytes_per_second':bandwidth,'ridge_flop_per_byte':ridge,'intensity_over_ridge':ratio,'roofline_regime':regime,'envelope_flops_per_second':roof,'envelope_attainment':rate/roof if rate is not None and roof and roof>0 else None,'projection_only':len(positive)>1 or not allknown,'causal_bottleneck':'unknown','causal_evidence':[]}
 points={}
 for p,n in ops.items():
  C=peaks[p];I=n/dram if n is not None and n>0 and dram is not None and dram>0 else None
  rate=n/dt if n is not None and n>0 and dt else None;ridge=C/bw if C is not None and C>0 and bw is not None and bw>0 else None
  ratio=I/ridge if I is not None and ridge else None
  regime='not_applicable'if n==0 else 'unclassified_missing_counter_or_ceiling'if ratio is None else 'near_ridge'if .8<=ratio<=1.25 else 'memory_side'if ratio<.8 else 'compute_side'
  roof=min(C,bw*I)if C is not None and bw is not None and I is not None else None
  if dram==0 and n is not None and n>0:regime='undefined_DRAM_intensity_zero_traffic'
  points[p]={'executed_scalar_flops':n,'modeled_clock_adjusted_peak_flops_per_second':C,'dram_arithmetic_intensity_flop_per_byte':I,'achieved_flops_per_second':rate,'ridge_flop_per_byte':ridge,'intensity_over_ridge':ratio,'roofline_regime':regime,'envelope_flops_per_second':roof,'envelope_attainment':rate/roof if rate is not None and roof and roof>0 else None,'projection_only':len(positive)>1 or not allknown,'causal_bottleneck':'unknown','causal_evidence':[]}
 if detected['fp16'] and not half_provenance:warnings.append('FP16 present but lane/FLOP convention unvalidated; no half FLOP point or peak')
 if detected['tensor'] and not tensor_valid:warnings.append('Tensor pipeline present; explicit512FLOP factor, sm70context and calibratedFP16input/FP32accumulation evidence required; tensor FLOPs/ceiling null')
 if tensor_flops_per_instruction is not None and not tensor_valid:warnings.append('Requested tensor convention not enabled: factor512/evidence/sm70context gate not satisfied')
 if not positive and allknown:warnings.append('No add/mul/FMA/tensor counted: may be integer, SFU, memory-only or eliminated work; requires source/SASS review, not automatic integer classification')
 return {'identity':g['identity'],'scope_context':context or {},'duration_seconds':dt,'precision_detection':detected,'precision_class':precision_class,'raw_scalar_predicated_on_thread_instruction_counts':raw,'tensor_pipe_instructions':tensor,'tensor_flops':tensor_flops,'tensor_flops_per_instruction':tensor_flops_per_instruction,'tensor_convention_evidence':tensor_convention_evidence,'tensor_convention_enabled':tensor_valid,'tensor_calibration_scope':'FP16inputs_FP32accumulation_HMMA884_sm70_only' if tensor_valid else None,'tensor_target_precision_scope':ctx.get('tensor_precision_mode','unreviewed'),'tensor_precision_projections':{tier:project(tensor_flops,tensor_peak,traffic,bandwidth,tier) for tier,traffic,bandwidth in [('dram',dram,bw),('l2',l2,l2bw)]},'scalar_precision_projections_by_memory_tier':{'dram':points,'l2':{p:project(n,peaks[p],l2,l2bw,'l2')for p,n in ops.items()}},'modeled_clock_adjusted_peak_L2_bandwidth_bytes_per_second':l2bw,'observed_integer_predicated_on_thread_instructions':metric('smsp__sass_thread_inst_executed_op_integer_pred_on.sum','inst'),'measured_utilization_pct':{'SM':metric('sm__throughput.avg.pct_of_peak_sustained_elapsed','%'),'DRAM':metric('dram__throughput.avg.pct_of_peak_sustained_elapsed','%')},'fp16_lane_factor':half_lanes,'fp16_convention_evidence':half_provenance,'dram_read_bytes':read,'dram_write_bytes':write,'dram_bytes':dram,'L2_transaction_bytes':metric('lts__t_bytes.sum','byte'),'modeled_clock_adjusted_peak_bandwidth_bytes_per_second':bw,'measured_DRAM_bytes_per_second':dram/dt if dram is not None and dt else None,'scalar_precision_projections':points,'causal_bottleneck':'unknown','warnings':warnings,'interpretation':'Per-kernel geometric roofline placement only. Conventional add/mul/FMA excludes integer/address/control/special-function/atomic costs. No equal-weight benchmark aggregation. Modeled peak ceiling is not empirically measured sustained performance.'}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--csv',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--context',type=pathlib.Path);ap.add_argument('--fp16-lanes',type=int,choices=[1,2]);ap.add_argument('--fp16-convention-evidence');ap.add_argument('--tensor-flops-per-instruction',type=float);ap.add_argument('--tensor-convention-evidence');a=ap.parse_args()
 if a.fp16_lanes is not None and not a.fp16_convention_evidence:ap.error('half lane factor requires evidence identifier/path')
 if a.tensor_flops_per_instruction is not None and not a.tensor_convention_evidence:ap.error('tensor factor requires explicit calibration evidence and sm70 context')
 gs,h=read_csv(a.csv);ctx=json.loads(a.context.read_text())if a.context else {}
 records=[derive(g,a.fp16_lanes,a.fp16_convention_evidence,ctx,a.tensor_flops_per_instruction,a.tensor_convention_evidence)for g in gs]
 report={'schema_version':1,'input_csv':str(a.csv),'input_sha256':h,'parser_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'kernel_count':len(records),'records':records,'benchmark_bottleneck':'not_aggregated','ceiling_kind':'modeled_theoretical_clock_adjusted_not_empirical_sustained'}
 a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'kernels':len(records),'output':str(a.output)}))
if __name__=='__main__':main()
