#!/usr/bin/env python3
"""CPU-only source/counter interpretation of twelve admitted scientific families.
No compiler, GPU, tests, benchmark or shared-data changes. Fresh /tmp output.
"""
import argparse,csv,gzip,hashlib,io,json,pathlib,collections
P=pathlib.Path
MODELS=('cuda','cuda_jl','ka','jacc')
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def raw_counts(raw):
 ls=raw.decode().splitlines();i=next(i for i,l in enumerate(ls)if l.startswith('"ID"')or l.startswith('ID,'));rows=list(csv.DictReader(io.StringIO('\n'.join(ls[i:]))));tot=collections.Counter();ids=set()
 for r in rows:
  if not r.get('ID','').isdigit():continue
  ident=tuple(r.get(k)for k in ['ID','Process ID','Context','Stream','Kernel Name']);ids.add(ident)
  if 'Metric Name'in r:
   if r['Metric Unit']=='inst'and r['Metric Name'].endswith('_pred_on.sum'):tot[r['Metric Name']]+=float(r['Metric Value'].replace(',',''))
  else:
   for k,v in r.items():
    if k.endswith('_pred_on.sum')and v:
     try:tot[k]+=float(v.replace(',',''))
     except ValueError:raise ValueError('Unknown instruction counter '+k)
 return tot,len(ids)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=P,default=P('/home/imo/HeCBench_julia'));ap.add_argument('--strict-audit',type=P);ap.add_argument('--output-root',type=P,required=True);a=ap.parse_args();root=a.source_root.resolve();out=a.output_root.absolute()
 if not out.resolve().is_relative_to(P('/tmp'))or out.resolve()==P('/tmp')or out.exists():raise ValueError('Fresh /tmp output required')
 inputs={}
 def bound(p,expected=None):
  p=P(p);p=p if p.is_absolute()else root/p;h=sha(p)
  if expected and h!=expected:raise ValueError('Input SHA mismatch '+str(p))
  inputs[str(p)]=h;return p
 auditpath=a.strict_audit or root/'ontology/studies/v100_roofline_537/retained_roi_aggregates/audit.json';audit=json.loads(bound(auditpath).read_text());selected=[r for r in audit['records']if r.get('accepted_complete_scientific_kernel_family')]
 if {(r['benchmark'],r['model'])for r in selected}!={(b,m)for b in ['triad','babelstream','jacobi']for m in MODELS}:raise ValueError('Twelve exact admitted families required')
 rows=[];sources={};missing_sass=[];compiler=[];rawfiles=[]
 for r in selected:
  b=r['benchmark'];m=r['model'];profile=P(r['selected_profile'][0]);profile=profile if profile.is_absolute()else root/profile;mp=bound(profile/'manifest.json');manifest=json.loads(mp.read_text());data={}
  for f in manifest['files']:
   p=bound(f['path'],f['compressed_sha256']);raw=gzip.decompress(p.read_bytes())
   if hashlib.sha256(raw).hexdigest()!=f['uncompressed_sha256']:raise ValueError('Frozen raw hash mismatch')
   if f['original_name']in ['plan.json','profile_provenance.json']:data[f['original_name']]=json.loads(raw)
   if f['original_name']=='counters.csv':counter=raw;rawfiles.append({'benchmark':b,'model':m,'path':str(p),'compressed_sha256':f['compressed_sha256'],'raw_sha256':f['uncompressed_sha256']})
  plan=data['plan.json'];prov=data['profile_provenance.json']
  if prov.get('status')!='profile_process_completed'or not prov.get('retained_window_expected_count_confirmed'):raise ValueError('Unconfirmed source-counted capture')
  count,n=raw_counts(counter)
  for required in ['fadd','fmul','ffma']:
   if 'smsp__sass_thread_inst_executed_op_'+required+'_pred_on.sum'not in count:raise ValueError('Missing arithmetic metric, not zero: '+required)
  op={k:count['smsp__sass_thread_inst_executed_op_'+k+'_pred_on.sum']for k in ['fadd','fmul','ffma']};F=op['fadd']+op['fmul']+2*op['ffma'];tot=r['totals']
  if F!=tot['fp32_executed_flops']or n!=tot['actual_scientific_launches']:raise ValueError('Strict audit/raw operation mismatch')
  rows.append({'benchmark':b,'model':m,'source_configuration':r['configuration_id'],'actual_launches':n,'fp32_fadd_instructions':op['fadd'],'fp32_fmul_instructions':op['fmul'],'fp32_ffma_instructions':op['ffma'],'weighted_fp32_FLOPs':F,'counted_arithmetic_instructions':sum(op.values()),'integer_predicated_on_instructions':count.get('smsp__sass_thread_inst_executed_op_integer_pred_on.sum'),'dram_read_bytes':tot['dram_read_bytes'],'dram_write_bytes':tot['dram_write_bytes'],'dram_bytes':tot['dram_bytes'],'L2_transaction_bytes':tot['L2_transaction_bytes'],'integrated_executed_fp32_FLOP_per_DRAM_byte':tot['fp32_flop_per_DRAM_byte'],'counter_sha256':prov['counters_sha256'],'selected_profile':str(profile)})
  for p,h in r['source_current_bindings'].items():
   q=bound(p,h)
   if q.suffix in ['.cu','.jl','.h','.cpp']:sources[str(q.resolve())]=h
  compiler.append({'benchmark':b,'model':m,'build_command':plan.get('build_command'),'profiler_version':prov.get('profiler_version'),'parser_sha256':next((x['uncompressed_sha256']for x in manifest['files']if x['original_name']=='parser_source.py'),None),'collector_sha256':prov.get('collector_sha256'),'declared_operation_calibration':prov.get('operation_calibration'),'before_after_source_recheck':prov.get('binding_recheck_after')})
  names=[f['original_name']for f in manifest['files']];missing_sass.append({'benchmark':b,'model':m,'selected_manifest_has_SASS_or_PTX':any(x.lower().endswith(('.sass','.ptx','.cubin'))for x in names),'selected_scratch_binary_retained':P(plan['scratch_cwd']).exists(),'scope':'No selected-instance disassembly retained in listed manifest; no recompile/JIT performed.'})
 backend=root/'src/_julia_jacc_env/Manifest.toml';bound(backend)
 installed=P('/home/imo/.julia/packages/JACC/NLI8A/ext/CUDAExt/CUDAExt.jl');srcj=bound(installed);sources[str(srcj)]=inputs[str(srcj)]
 atomix=bound('/home/imo/.julia/packages/JACC/NLI8A/src/JACC.jl');sources[str(atomix)]=inputs[str(atomix)]
 by={(r['benchmark'],r['model']):r for r in rows};N=2048;steps=20;interior=(N-2)**2;blocks=(N//16)**2
 arithmetic={'Jacobi':{'N':N,'steps':steps,'interior_updates_per_step':interior,'core_per_update':{'fadd_or_subtract':4,'fmul':2},'core_fp32_FLOPs':steps*interior*6,'first_reduction_all_threads_4adds':steps*N*N*4,'second_reduction_16threads_per_block_4adds':steps*blocks*16*4,'extra_native_FADD':steps*(N*N*4+blocks*16*4),'native_total':steps*(interior*6+N*N*4+blocks*16*4),'native_atomic_calls_source_unconditional_block_count':steps*blocks,'Julia_atomic_calls':'data dependent: error>0 for each interior point; exact dynamic count not collected','native_reads_source_model_per_step':N*N+4*(N//16-1)*(N//16)*16,'Julia_direct_scalar_input_loads_source_model_per_step':interior*5,'source_load_counts_not_DRAM_transactions':True},'STREAM_dot':{'N':262144,'repeats':10,'groups':256,'threads_per_group':256,'dot_product_accumulation_FLOPs':2*262144*10,'block_tree_reduction_FADD':256*255*10,'total_fp32_FLOPs':2*262144*10+256*255*10,'JACC_source_template_bytes_if_copy_emitted_every_block':256*4*256*10,'JACC_template_copy_destination':'dynamic shared memory, not a source-declared global-memory output','optimizer_emission_not_inferred_from_source_copy':True},'Triad':{'N':4194304,'repeats':100,'total_useful_and_executed_fp32_FLOPs':2*4194304*100,'native_instruction_lowering':'FFMA counter positive; separate FADD/FMUL zero','Julia_instruction_lowering':'FADD and FMUL counters positive; FFMA zero'}}
 if arithmetic['Jacobi']['core_fp32_FLOPs']!=by['jacobi','cuda_jl']['weighted_fp32_FLOPs']or arithmetic['Jacobi']['native_total']!=by['jacobi','cuda']['weighted_fp32_FLOPs']:raise ValueError('Jacobi source census does not reconcile raw counts')
 ratios=[]
 for b in ['triad','babelstream','jacobi']:
  for baseline in ['cuda','cuda_jl','ka']:
   j=by[b,'jacc'];o=by[b,baseline];ratios.append({'benchmark':b,'target':'jacc','baseline':baseline,'DRAM_total_ratio':j['dram_bytes']/o['dram_bytes'],'DRAM_read_ratio':j['dram_read_bytes']/o['dram_read_bytes'],'DRAM_write_difference_bytes':j['dram_write_bytes']-o['dram_write_bytes'],'DRAM_total_difference_bytes':j['dram_bytes']-o['dram_bytes'],'L2_total_ratio':j['L2_transaction_bytes']/o['L2_transaction_bytes'],'interpretation':'Observed cold-replay traffic ratio, not a causal effect or ordinary host ROI throughput ratio'})
 for p,h in inputs.items():
  if sha(p)!=h:raise ValueError('Input changed during read-only analysis')
 out.mkdir(parents=True);dump(out/'analysis.json',{'schema_version':1,'input_sha256':inputs,'strict_audit':str(auditpath),'operation_counts':rows,'source_operation_models':arithmetic,'JACC_traffic_comparisons':ratios,'compiler_capture_evidence':compiler,'SASS_availability':missing_sass,'current_installed_JACC_source_scope':'Current installed source and captured pinned environment hashed; not a selected-instance SASS capture.','scientific_limitations':['Executed add/mul/FMA counts exclude atomic/SFU/integer costs.','DRAM bytes are hardware interval transactions, not source store instruction counts.','Source per-point atomics vs native block reduction is a known algorithm mapping adaptation.','JACC higher DRAM writes are observed; source/SASS evidence here does not establish their cause.','No whole elapsedROI timing or causal bottleneck join.']})
 with(out/'operation_traffic_table.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with(out/'JACC_traffic_comparisons.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(ratios[0]));w.writeheader();w.writerows(ratios)
 excerpts=[]
 for path,h in sorted(sources.items()):
  p=P(path);lines=p.read_text().splitlines();keep=[]
  if path.endswith('/jacobi-cuda/main.cu'):lo,hi=47,126
  elif path.endswith('/CUDAExt/CUDAExt.jl'):lo,hi=450,468
  elif 'babelstream-cuda/main.cu'in path:lo,hi=198,225
  elif p.name=='main.jl'and('jacobi-julia'in path or'babelstream-julia'in path):lo,hi=1,min(len(lines),58)
  else:continue
  excerpts.append({'path':path,'sha256':h,'first_line':lo,'last_line':hi,'text':'\n'.join(str(i)+': '+lines[i-1]for i in range(lo,min(hi,len(lines))+1))})
 lines=installed.read_text().splitlines();excerpts.append({'path':str(installed),'sha256':inputs[str(installed)],'first_line':598,'last_line':609,'text':'\n'.join(str(i)+': '+lines[i-1]for i in range(598,610))});dump(out/'source_excerpts.json',excerpts);dump(out/'counter_bindings.json',rawfiles);(out/'analyzer_source_snapshot.py').write_bytes(P(__file__).read_bytes());print(json.dumps({'output':str(out),'families':len(rows),'Jacobi_extra_FADD':arithmetic['Jacobi']['extra_native_FADD'],'Jacobi_core_FP32_FLOPs':arithmetic['Jacobi']['core_fp32_FLOPs']}))
if __name__=='__main__':main()
